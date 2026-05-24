# Chess AI Training Pipeline

## Architecture

**AlphaZero-style neural network** with supervised pre-training on master games, optionally fine-tuned via self-play.

### Model: `training/model.py`
- **Input**: 18-plane 8×8 board representation (12 piece planes + 4 castling + 1 en passant + 1 side to move)
- **Backbone**: 6 residual blocks, 128 filters — deliberately small for browser ONNX inference
- **Policy head**: 4,672 outputs (AlphaZero 8×8×73 move encoding)
- **Value head**: scalar in [-1, 1] predicting win probability
- **Parameters**: ~11.4M

### Move Encoding: `training/encoding.py`
Uses the AlphaZero 8×8×73 scheme:
- 56 queen-type moves (7 distances × 8 directions)
- 8 knight moves
- 9 underpromotions (3 directions × 3 piece types)

Board is always encoded from the perspective of the side to move (flipped for black).

---

## Training Approaches

### 1. Supervised Pre-Training (Recommended First Step)

**File**: `training/supervised.py`

Trains the network to predict master-level moves and game outcomes from real games.

**Why this approach?**
- AlphaZero self-play from scratch requires enormous compute (Google used 5,000 TPUs)
- With a single GPU, self-play produces only draws for hundreds of iterations because the random model can't force wins
- Supervised pre-training gives the model strong chess knowledge in ~30 minutes
- The pre-trained model can then be fine-tuned with self-play

**Data source**: Lichess Elite Database (2500+ vs 2300+ rated players)
- URL: https://database.nikonoel.fr/
- ~300k games per month, ~65 MB compressed
- Free, Creative Commons license

**Usage**:
```bash
# Download and generate training data
python download_data.py

# Train on real master games
python -m training.supervised \
  --pgn data/lichess_elite_2025-01.pgn \
  --max-games 50000 \
  --sample-per-game 12 \
  --epochs 15 \
  --min-elo 2300 \
  --export

# Train on synthetic games (fallback if no PGN available)
python -m training.supervised --download --epochs 10 --export
```

**What it learns**:
- **Policy**: Cross-entropy loss against the master's move → predicts strong moves
- **Value**: MSE loss against game result → evaluates positions

**Expected results** (50k real games, 15 epochs):
- Validation move accuracy: 25-35%
- Meaningful value predictions for material imbalances
- Strong opening play (knows main lines)

### 2. Self-Play Reinforcement Learning

**Files**: `training/self_play.py`, `training/mcts.py`, `training/train.py`

AlphaZero-style training loop: self-play → collect training examples → train → evaluate.

**Why it's slow on consumer hardware**:
- MCTS tree search is CPU-bound (Python loop overhead)
- GPU utilization is <5% — the bottleneck is tree traversal, not neural network inference
- An untrained model plays randomly, so all games end in draws (no learning signal)
- Resignation logic helps but only after the value head learns to distinguish good/bad positions

**Optimizations applied**:
- Batched MCTS leaf evaluation (batch_size=32) — collects multiple leaves before GPU call
- Resignation threshold (-0.9 for 5 consecutive moves) — creates decisive games
- Parallel game batching — multiple games advance simultaneously

**Usage**:
```bash
# Quick test (3 iterations, ~10 min)
python -m training.run_training --quick

# Full training with resume
python -m training.run_training --games 30 --simulations 50 --iterations 50 --resume

# Export best model to ONNX
python -m training.run_training --export-only
```

**Recommendation**: Always start with supervised pre-training, then optionally fine-tune with self-play using `--resume`.

---

## ONNX Export: `training/export_onnx.py`

Converts PyTorch checkpoint to ONNX format (43 MB) for browser inference via ONNX Runtime Web.

```bash
python -m training.run_training --export-only
```

Output: `game/static/engine/model.onnx`

---

## File Structure

```
training/
├── __init__.py
├── model.py          # AlphaZeroNet architecture
├── encoding.py       # Board ↔ tensor, move ↔ index conversion
├── mcts.py           # Monte Carlo Tree Search with batched GPU evaluation
├── self_play.py      # Self-play game generation with resignation
├── train.py          # Self-play training loop (RL)
├── supervised.py     # Supervised pre-training on master games
├── run_training.py   # CLI entry point for self-play training
├── export_onnx.py    # PyTorch → ONNX export
```

```
data/
├── master_games.pgn         # Synthetic training games (auto-generated)
├── lichess_elite_2025-01.pgn # Real master games (downloaded)
├── lichess_elite.zip         # Compressed download
```

```
checkpoints/
├── best.pt    # Best model (by validation loss or evaluation win rate)
├── latest.pt  # Most recent checkpoint
```

---

## Hardware Notes

- **GPU**: NVIDIA RTX 3060 Laptop (6 GB VRAM) — sufficient for this model size
- **CPU**: 20 cores — MCTS is CPU-bound, more cores help but Python GIL limits true parallelism
- **Multiprocessing**: Attempted but CUDA + Windows `spawn` causes hangs. Sequential batched approach is used instead.
- **Training speed**: ~7s per supervised epoch (569k samples), ~4 min per self-play iteration (5 games × 50 sims)

## Key Lessons

1. **Self-play from scratch is impractical on consumer hardware** — the model needs millions of games to emerge from random play. Google's AlphaZero used 5,000 TPUs for 9 hours.

2. **Supervised pre-training is the practical path** — 50k master games give strong chess knowledge in 30 minutes on a single GPU.

3. **GPU utilization in MCTS is inherently low** — tree traversal is serial Python code. The GPU is idle 95%+ of the time during self-play. This is a fundamental limitation of Python MCTS, not a configuration issue.

4. **Resignation is essential** — without it, untrained models play 120+ move games that all end in draws, providing zero learning signal for the value head.

5. **Board perspective flipping matters** — the network always sees the board from the current player's perspective. Forgetting to flip for black moves causes the model to learn inverted evaluations.

---

## Browser Inference Pipeline

### ONNX Runtime Web + MCTS (client-side)

The trained model runs entirely in the browser via a Web Worker:

**File**: `game/static/engine/engine-worker.js`

- Loads `model_q8.onnx` (10.9 MB quantized) via ONNX Runtime Web
- Implements full MCTS with Dirichlet noise at root
- Board-to-tensor conversion mirrors `training/encoding.py` (18×8×8 planes)
- Move indexing uses the same AlphaZero 8×8×73 scheme
- Level configs: Beginner (50 sims), Intermediate (150), Advanced (400)

**ONNX Quantization**:
```bash
python -m training.export_onnx         # Full model (43 MB)
python -m training.export_onnx --quantize  # INT8 quantized (10.9 MB)
```

The quantized model (`model_q8.onnx`) is 4× smaller with minimal accuracy loss, critical for browser loading time.

### WASM Files

ONNX Runtime Web requires WASM binaries served from the same origin:
- `ort-wasm-simd-threaded.jsep.mjs` — JavaScript glue code
- `ort-wasm-simd-threaded.jsep.wasm` — compiled WASM binary

These are stored in `game/static/engine/` and served as static files.

---

## Stockfish.js Integration (Position Evaluation)

**File**: `game/static/engine/stockfish.js` (Stockfish v10.0.2, ~1.5 MB)

Used purely for position evaluation (not move generation). Runs as a separate Web Worker alongside the neural net engine.

**Purpose**: Provides accurate centipawn evaluations for move classification. The neural net's value head gives relative evaluations but isn't precise enough for classifying moves as Good/Inaccuracy/Mistake/Blunder.

**Evaluation Flow**:
1. After AI moves → Stockfish evaluates at depth 12 → stores as `sfEvalBefore` (baseline)
2. After player moves → Stockfish evaluates at depth 12 → computes `cpLoss = baseline - (-eval_after)`
3. `cpLoss` classified via thresholds: Brilliant (≤-1.0), Good (≤0.5), OK (≤1.0), Inaccuracy (≤2.0), Mistake (≤4.0), Blunder (>4.0)

**Queue-based eval system**: Uses `sfEvalQueue` array to handle race conditions between baseline and post-move evaluations. On player move, any in-progress baseline search is cancelled via UCI `stop` command.

**Mate score handling**: Scores are clamped to avoid absurd centipawn values (e.g., mate-in-5 showing as 4612cp loss).
