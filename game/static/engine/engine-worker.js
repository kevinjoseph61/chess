/**
 * Chess Engine Web Worker
 * 
 * Runs AlphaZero-style MCTS + neural network inference entirely in the browser
 * using ONNX Runtime Web (WebAssembly backend).
 *
 * Messages IN:
 *   { type: 'init' }                         — load ONNX model
 *   { type: 'search', fen, level }           — find best move for position
 *
 * Messages OUT:
 *   { type: 'ready' }                        — model loaded
 *   { type: 'move', move, eval }             — best move found
 *   { type: 'error', message }               — error occurred
 *   { type: 'status', message }              — status update
 */

// Import ONNX Runtime Web
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js');

// Point WASM files to CDN
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// ============================================================
// Board representation (minimal chess logic for encoding)
// ============================================================

const PIECE_TYPES = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
const WHITE = true;
const BLACK = false;

// Square indices: a1=0, b1=1, ..., h8=63
function sq(file, rank) { return rank * 8 + file; }
function sqRank(s) { return s >> 3; }
function sqFile(s) { return s & 7; }

/**
 * Parse FEN into a board state object.
 */
function parseFEN(fen) {
  const parts = fen.split(' ');
  const pieces = new Array(64).fill(null); // {type, color}
  const rows = parts[0].split('/');
  
  for (let r = 0; r < 8; r++) {
    let col = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') {
        col += parseInt(ch);
      } else {
        const color = ch === ch.toUpperCase() ? WHITE : BLACK;
        const type = PIECE_TYPES[ch.toLowerCase()];
        pieces[sq(col, 7 - r)] = { type, color };
        col++;
      }
    }
  }
  
  const turn = parts[1] === 'w' ? WHITE : BLACK;
  
  // Castling rights as bitmask matching python-chess BB constants
  let castling = 0;
  const castleStr = parts[2] || '-';
  if (castleStr.includes('K')) castling |= (1 << sq(7, 0)); // BB_H1
  if (castleStr.includes('Q')) castling |= (1 << sq(0, 0)); // BB_A1
  if (castleStr.includes('k')) castling |= (1 << sq(7, 7)); // BB_H8 — use separate flag
  if (castleStr.includes('q')) castling |= (1 << sq(0, 7)); // BB_A8
  // Actually, we just need boolean flags:
  const castleK = castleStr.includes('K');
  const castleQ = castleStr.includes('Q');
  const castlek = castleStr.includes('k');
  const castleq = castleStr.includes('q');
  
  // En passant square
  let epSquare = null;
  if (parts[3] && parts[3] !== '-') {
    const file = parts[3].charCodeAt(0) - 97; // 'a' = 0
    const rank = parseInt(parts[3][1]) - 1;
    epSquare = sq(file, rank);
  }
  
  return { pieces, turn, castleK, castleQ, castlek, castleq, epSquare };
}

/**
 * Encode board state as 18x8x8 Float32Array matching Python board_to_tensor().
 */
function boardToTensor(boardState) {
  const tensor = new Float32Array(18 * 8 * 8);
  const flip = !boardState.turn; // flip if black to move
  
  // Piece plane mapping: [pieceType][color] -> plane
  // Same as Python PIECE_PLANES
  const planeMap = {
    1: { true: 0, false: 6 },   // pawn
    2: { true: 1, false: 7 },   // knight
    3: { true: 2, false: 8 },   // bishop
    4: { true: 3, false: 9 },   // rook
    5: { true: 4, false: 10 },  // queen
    6: { true: 5, false: 11 },  // king
  };
  
  for (let s = 0; s < 64; s++) {
    const piece = boardState.pieces[s];
    if (!piece) continue;
    
    let row = sqRank(s);
    let col = sqFile(s);
    let color = piece.color;
    
    if (flip) {
      row = 7 - row;
      color = !color;
    }
    
    const plane = planeMap[piece.type][color];
    tensor[plane * 64 + row * 8 + col] = 1.0;
  }
  
  // Castling rights (planes 12-15)
  if (boardState.castleK) for (let i = 0; i < 64; i++) tensor[12 * 64 + i] = 1.0;
  if (boardState.castleQ) for (let i = 0; i < 64; i++) tensor[13 * 64 + i] = 1.0;
  if (boardState.castlek) for (let i = 0; i < 64; i++) tensor[14 * 64 + i] = 1.0;
  if (boardState.castleq) for (let i = 0; i < 64; i++) tensor[15 * 64 + i] = 1.0;
  
  // En passant (plane 16)
  if (boardState.epSquare !== null) {
    let epRow = sqRank(boardState.epSquare);
    let epCol = sqFile(boardState.epSquare);
    if (flip) epRow = 7 - epRow;
    tensor[16 * 64 + epRow * 8 + epCol] = 1.0;
  }
  
  // Side to move (plane 17): 1 if white to move
  if (boardState.turn === WHITE) {
    for (let i = 0; i < 64; i++) tensor[17 * 64 + i] = 1.0;
  }
  
  return tensor;
}


// ============================================================
// Move encoding (AlphaZero 8x8x73 scheme)
// ============================================================

// Queen directions: N, NE, E, SE, S, SW, W, NW
const QUEEN_DIRS = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]];

// Knight moves
const KNIGHT_MOVES = [[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1]];

// Underpromotion directions (from 7th rank pawn perspective)
const PROMO_DIRS = [[-1,-1],[-1,0],[-1,1]]; // left capture, forward, right capture
// Promo pieces: knight=0, bishop=1, rook=2 (queen = default queen move encoding)

/**
 * Encode a move (from_sq, to_sq, promotion) to flat index [0, 4671].
 * promotion: null, 'n', 'b', 'r', 'q'
 */
function moveToIndex(fromSq, toSq, promotion, flip) {
  let fromRow = sqRank(fromSq);
  let fromCol = sqFile(fromSq);
  let toRow = sqRank(toSq);
  let toCol = sqFile(toSq);
  
  if (flip) {
    fromRow = 7 - fromRow;
    toRow = 7 - toRow;
  }
  
  const dr = toRow - fromRow;
  const dc = toCol - fromCol;
  
  // Check underpromotion (knight, bishop, rook — NOT queen)
  if (promotion && promotion !== 'q') {
    const promoIdx = { n: 0, b: 1, r: 2 }[promotion];
    if (promoIdx !== undefined) {
      for (let dirIdx = 0; dirIdx < PROMO_DIRS.length; dirIdx++) {
        if (dr === PROMO_DIRS[dirIdx][0] && dc === PROMO_DIRS[dirIdx][1]) {
          const plane = 64 + dirIdx * 3 + promoIdx;
          return fromRow * 8 * 73 + fromCol * 73 + plane;
        }
      }
    }
  }
  
  // Check knight move
  for (let i = 0; i < KNIGHT_MOVES.length; i++) {
    if (dr === KNIGHT_MOVES[i][0] && dc === KNIGHT_MOVES[i][1]) {
      const plane = 56 + i;
      return fromRow * 8 * 73 + fromCol * 73 + plane;
    }
  }
  
  // Queen-type move (includes queen promotion)
  const normDr = dr === 0 ? 0 : dr / Math.abs(dr);
  const normDc = dc === 0 ? 0 : dc / Math.abs(dc);
  const distance = Math.max(Math.abs(dr), Math.abs(dc));
  
  for (let dirIdx = 0; dirIdx < QUEEN_DIRS.length; dirIdx++) {
    if (normDr === QUEEN_DIRS[dirIdx][0] && normDc === QUEEN_DIRS[dirIdx][1]) {
      const plane = dirIdx * 7 + (distance - 1);
      return fromRow * 8 * 73 + fromCol * 73 + plane;
    }
  }
  
  return -1; // shouldn't happen
}

/**
 * Convert a UCI move string (e.g., "e2e4", "a7a8q") to {from, to, promotion}.
 */
function parseUCI(uci) {
  const fromFile = uci.charCodeAt(0) - 97;
  const fromRank = parseInt(uci[1]) - 1;
  const toFile = uci.charCodeAt(2) - 97;
  const toRank = parseInt(uci[3]) - 1;
  const promotion = uci.length > 4 ? uci[4] : null;
  return {
    from: sq(fromFile, fromRank),
    to: sq(toFile, toRank),
    promotion
  };
}

/**
 * Convert internal square to UCI notation.
 */
function sqToUCI(s) {
  return String.fromCharCode(97 + sqFile(s)) + (sqRank(s) + 1);
}

/**
 * Get all legal moves from a chess.js game and encode them.
 * Returns array of { uci, from, to, promotion, index }.
 */
function getLegalMoves(game, flip) {
  const moves = game.moves({ verbose: true });
  return moves.map(m => {
    const fromFile = m.from.charCodeAt(0) - 97;
    const fromRank = parseInt(m.from[1]) - 1;
    const toFile = m.to.charCodeAt(0) - 97;
    const toRank = parseInt(m.to[1]) - 1;
    const fromSq = sq(fromFile, fromRank);
    const toSq = sq(toFile, toRank);
    const promo = m.promotion || null;
    const index = moveToIndex(fromSq, toSq, promo, flip);
    const uci = m.from + m.to + (promo || '');
    return { uci, from: fromSq, to: toSq, promotion: promo, index, san: m.san };
  });
}


// ============================================================
// MCTS (Monte Carlo Tree Search)
// ============================================================

class MCTSNode {
  constructor(parent, priorP, move) {
    this.parent = parent;
    this.move = move;      // { uci, san } that led to this node
    this.children = [];
    this.visitCount = 0;
    this.totalValue = 0;
    this.priorP = priorP;
    this.isExpanded = false;
  }
  
  get qValue() {
    return this.visitCount > 0 ? this.totalValue / this.visitCount : 0;
  }
  
  ucbScore(parentVisits, cPuct) {
    const exploration = cPuct * this.priorP * Math.sqrt(parentVisits) / (1 + this.visitCount);
    return this.qValue + exploration;
  }
  
  bestChild(cPuct) {
    let best = null;
    let bestScore = -Infinity;
    for (const child of this.children) {
      const score = child.ucbScore(this.visitCount, cPuct);
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }
  
  expand(legalMoves, policy) {
    // legalMoves: array of { uci, san, index }
    // policy: Float32Array of size 4672 (raw logits after masking + softmax)
    for (const m of legalMoves) {
      const prior = policy[m.index] || 1e-8;
      this.children.push(new MCTSNode(this, prior, { uci: m.uci, san: m.san }));
    }
    this.isExpanded = true;
  }
  
  backpropagate(value) {
    this.visitCount++;
    this.totalValue += value;
    if (this.parent) {
      this.parent.backpropagate(-value); // negate for opponent
    }
  }
}


// ============================================================
// Engine
// ============================================================

let session = null;
const C_PUCT = 1.5;

// Simulation counts per level
const LEVEL_SIMS = {
  1: 50,   // Beginner
  2: 150,  // Intermediate
  3: 400,  // Advanced
};

async function initModel() {
  try {
    postMessage({ type: 'status', message: 'Loading neural network...' });
    
    // Configure ONNX Runtime for WASM
    ort.env.wasm.numThreads = 1;
    
    session = await ort.InferenceSession.create('/static/engine/model_q8.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    
    postMessage({ type: 'status', message: 'Neural network loaded!' });
    postMessage({ type: 'ready' });
  } catch (e) {
    postMessage({ type: 'error', message: 'Failed to load model: ' + e.message });
  }
}

/**
 * Run neural network inference on a board tensor.
 * Returns { policy: Float32Array(4672), value: number }.
 */
async function evaluate(boardTensor) {
  const inputTensor = new ort.Tensor('float32', boardTensor, [1, 18, 8, 8]);
  const results = await session.run({ board_state: inputTensor });
  
  const policy = results.policy.data;  // Float32Array(4672)
  const value = results.value.data[0]; // scalar
  
  return { policy, value };
}

/**
 * Apply softmax to policy logits, masked to legal moves only.
 */
function maskedSoftmax(logits, legalMoves) {
  const result = new Float32Array(logits.length);
  
  // Find max for numerical stability
  let maxVal = -Infinity;
  for (const m of legalMoves) {
    if (logits[m.index] > maxVal) maxVal = logits[m.index];
  }
  
  // Compute exp and sum
  let sumExp = 0;
  for (const m of legalMoves) {
    const exp = Math.exp(logits[m.index] - maxVal);
    result[m.index] = exp;
    sumExp += exp;
  }
  
  // Normalize
  if (sumExp > 0) {
    for (const m of legalMoves) {
      result[m.index] /= sumExp;
    }
  }
  
  return result;
}

/**
 * Run MCTS search from a given FEN position.
 * Uses chess.js for move generation (imported via importScripts).
 * 
 * Returns { move: string (UCI), eval: number }.
 */
async function search(fen, numSimulations) {
  // We need chess.js for legal move generation
  // Create game from FEN
  const game = new Chess(fen);
  const boardState = parseFEN(fen);
  const flip = !boardState.turn;
  
  // Get legal moves
  const legalMoves = getLegalMoves(game, flip);
  if (legalMoves.length === 0) {
    return { move: null, eval: 0 };
  }
  if (legalMoves.length === 1) {
    return { move: legalMoves[0].uci, eval: 0 };
  }
  
  // Initial evaluation
  const boardTensor = boardToTensor(boardState);
  const { policy: rawPolicy, value: rootValue } = await evaluate(boardTensor);
  const policy = maskedSoftmax(rawPolicy, legalMoves);
  
  // Create root node and expand
  const root = new MCTSNode(null, 1.0, null);
  root.expand(legalMoves, policy);
  
  // Add Dirichlet noise to root
  const alpha = 0.3;
  const noise = dirichletNoise(legalMoves.length, alpha);
  const epsilon = 0.25;
  for (let i = 0; i < root.children.length; i++) {
    root.children[i].priorP = (1 - epsilon) * root.children[i].priorP + epsilon * noise[i];
  }
  
  // Run simulations
  for (let sim = 0; sim < numSimulations; sim++) {
    // Selection: traverse tree using UCB
    let node = root;
    const gameCopy = new Chess(fen);
    
    while (node.isExpanded && node.children.length > 0) {
      node = node.bestChild(C_PUCT);
      gameCopy.move(node.move.san);
    }
    
    // Check terminal state
    if (gameCopy.game_over()) {
      let value = 0;
      if (gameCopy.in_checkmate()) {
        value = -1; // current side to move is checkmated
      }
      node.backpropagate(value);
      continue;
    }
    
    // Expansion: evaluate and expand
    const simFen = gameCopy.fen();
    const simBoardState = parseFEN(simFen);
    const simFlip = !simBoardState.turn;
    const simLegalMoves = getLegalMoves(gameCopy, simFlip);
    
    const simTensor = boardToTensor(simBoardState);
    const { policy: simRawPolicy, value: simValue } = await evaluate(simTensor);
    const simPolicy = maskedSoftmax(simRawPolicy, simLegalMoves);
    
    node.expand(simLegalMoves, simPolicy);
    
    // Backpropagate: negate value because it's from the perspective of the side to move
    node.backpropagate(-simValue);
  }
  
  // Select best move by visit count (temperature = 0 for max strength)
  let bestChild = null;
  let bestVisits = -1;
  for (const child of root.children) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestChild = child;
    }
  }
  
  // Calculate eval: root Q-value from the current side's perspective
  const evalScore = bestChild ? bestChild.qValue : 0;
  
  return {
    move: bestChild.move.uci,
    eval: evalScore,
  };
}

/**
 * Generate Dirichlet noise for MCTS root exploration.
 */
function dirichletNoise(n, alpha) {
  // Simple approximation using Gamma distribution
  const samples = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    // Gamma(alpha, 1) approximation using Marsaglia and Tsang's method
    samples[i] = gammaSample(alpha);
    sum += samples[i];
  }
  for (let i = 0; i < n; i++) {
    samples[i] /= sum;
  }
  return samples;
}

function gammaSample(alpha) {
  // For alpha < 1, use the algorithm from Ahrens-Dieter
  if (alpha < 1) {
    const u = Math.random();
    return gammaSample(1 + alpha) * Math.pow(u, 1 / alpha);
  }
  
  // Marsaglia and Tsang's method for alpha >= 1
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  
  while (true) {
    let x, v;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    
    v = v * v * v;
    const u = Math.random();
    
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function randn() {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}


// ============================================================
// Message handling
// ============================================================

// Import chess.js for move generation
importScripts('https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.2/chess.js');

onmessage = async function(e) {
  const msg = e.data;
  
  switch (msg.type) {
    case 'init':
      await initModel();
      break;
      
    case 'search':
      if (!session) {
        postMessage({ type: 'error', message: 'Model not loaded yet' });
        return;
      }
      try {
        const level = msg.level || 1;
        const numSims = LEVEL_SIMS[level] || 50;
        const result = await search(msg.fen, numSims);
        postMessage({ type: 'move', move: result.move, eval: result.eval });
      } catch (e) {
        postMessage({ type: 'error', message: 'Search error: ' + e.message });
      }
      break;
  }
};
