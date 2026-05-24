// Stockfish evaluation worker wrapper
// Loads Stockfish WASM as a sub-worker and provides position evaluation via UCI protocol

var stockfish = null;
var isReady = false;
var outputBuffer = [];
var searchDepth = 12;

// Create Stockfish as a sub-worker from CDN
try {
  stockfish = new Worker('https://cdn.jsdelivr.net/npm/stockfish.js@10.0.2/stockfish.js');
} catch(e) {
  // If cross-origin worker fails, try fetching as blob
  var xhr = new XMLHttpRequest();
  xhr.open('GET', 'https://cdn.jsdelivr.net/npm/stockfish.js@10.0.2/stockfish.js', false);
  xhr.send();
  var blob = new Blob([xhr.responseText], { type: 'application/javascript' });
  stockfish = new Worker(URL.createObjectURL(blob));
}

stockfish.onmessage = function(event) {
  handleStockfishOutput(typeof event.data === 'string' ? event.data : String(event.data));
};

self.onmessage = function(e) {
  var data = e.data;

  if (data.type === 'init') {
    stockfish.postMessage('uci');
    stockfish.postMessage('setoption name Hash value 16');
    stockfish.postMessage('isready');
  }
  else if (data.type === 'eval') {
    if (!isReady) {
      self.postMessage({ type: 'error', message: 'Stockfish not ready' });
      return;
    }
    stockfish.postMessage('ucinewgame');
    stockfish.postMessage('position fen ' + data.fen);
    stockfish.postMessage('go depth ' + (data.depth || searchDepth));
  }
  else if (data.type === 'setDepth') {
    searchDepth = data.depth;
  }
};

function handleStockfishOutput(line) {
  if (line === 'readyok') {
    isReady = true;
    self.postMessage({ type: 'ready' });
    return;
  }

  if (line.indexOf('bestmove') === 0) {
    // Search complete - extract the last score from buffer
    var lastScore = null;
    var lastPV = '';
    for (var i = outputBuffer.length - 1; i >= 0; i--) {
      var infoLine = outputBuffer[i];
      if (infoLine.indexOf('info depth') === 0 && infoLine.indexOf(' score ') !== -1) {
        // Parse score
        var scoreMatch = infoLine.match(/score (cp|mate) (-?\d+)/);
        if (scoreMatch) {
          if (scoreMatch[1] === 'cp') {
            lastScore = parseInt(scoreMatch[2]) / 100.0; // Convert centipawns to pawns
          } else {
            // Mate score: use large value
            var mateIn = parseInt(scoreMatch[2]);
            lastScore = mateIn > 0 ? 100 : -100;
          }
        }
        // Parse PV (principal variation)
        var pvMatch = infoLine.match(/ pv (.+)/);
        if (pvMatch) lastPV = pvMatch[1];
        break;
      }
    }

    var bestMove = line.split(' ')[1];
    self.postMessage({
      type: 'eval',
      score: lastScore,
      bestMove: bestMove,
      pv: lastPV
    });
    outputBuffer = [];
    return;
  }

  // Buffer info lines during search
  if (line.indexOf('info ') === 0) {
    outputBuffer.push(line);
  }
}
