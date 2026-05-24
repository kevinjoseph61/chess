# Chess Gang ♟️

A real-time chess platform with an AlphaZero-style AI engine and live LLM coaching — all running in the browser.

> **Looking for the original version?** The classic Chess Gang (Django + Redis multiplayer) is available on the [`master` branch](../../tree/master).

![Lobby](docs/screenshots/lobby.png)

## Features

### 🤖 AI-Powered Single Player
- **AlphaZero-style engine** — Neural network + MCTS running entirely in-browser via ONNX Runtime Web
- **Three difficulty levels** — Beginner (50 sims), Intermediate (150), Advanced (400)
- **Zero server load** — AI runs as a Web Worker on the client

### 🧑‍🏫 Real-Time AI Coach
- **Per-move evaluation** — Every move gets classified: ★ Brilliant, ✓ Good, OK, ?! Inaccuracy, ? Mistake, ?? Blunder
- **Stockfish-powered analysis** — Accurate centipawn evaluation via Stockfish.js (depth 12)
- **Live LLM coaching** — Natural language tips after each move using Llama 3.3 70B (Groq free tier)
- **Eval bar** — Visual evaluation bar showing position advantage
- **Score tracking** — Cumulative score based on move quality

![AI Coach in action](docs/screenshots/coaching.png)

### ⚔️ Real-Time Multiplayer
- **WebSocket-based** — Instant move synchronization via Django Channels + Redis
- **Public & private games** — Create open lobbies or invite specific opponents
- **Connection handling** — Auto-reconnect with opponent online/offline status

### 🎨 Modern Dark Theme
- **Bootstrap 5.3** dark theme across all pages
- **Responsive design** — Works on desktop and mobile
- **9 redesigned templates** — Lobby, game, create, ongoing, completed, login, signup

![Login](docs/screenshots/login.png)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Redis (for multiplayer — `docker run -p 6379:6379 -d redis:5`)
- [Groq API key](https://console.groq.com/) (free — for coaching feature)

### Setup

```bash
# Clone and install
git clone https://github.com/your-username/chess.git
cd chess
pipenv install

# Configure environment
echo "GROQ_API_KEY=your_key_here" > .env

# Run server
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` — no account needed to browse, register to play.

### Single Player (no Redis needed)
The single player mode with AI coaching works without Redis. Just start the server and navigate to **Single Player**.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and component breakdown.

**Key components:**
- **Browser**: Chess.js + Chessboard.js (UI), Engine Worker (MCTS + ONNX), Stockfish Worker (eval)
- **Server**: Django 5.2 + Daphne (ASGI), Channels (WebSocket), Groq (LLM)
- **Training**: PyTorch → ONNX export, supervised + self-play pipeline

```
Browser (Client)                          Server (Django)
┌──────────────────────┐                 ┌──────────────┐
│ Engine Worker (ONNX) │                 │ /api/coach/  │──→ Groq LLM
│ Stockfish Worker     │──HTTP POST──→   │ /api/analyze/│
│ Chess UI (Board.js)  │                 │ WebSocket    │──→ Redis
└──────────────────────┘                 └──────────────┘
```

## Design Decisions

See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for detailed rationale on:
- Why AI runs in-browser (zero server cost)
- Dual worker architecture (neural net + Stockfish)
- Supervised pre-training vs pure self-play
- ONNX INT8 quantization (43 MB → 10.9 MB)
- Groq free tier for coaching
- Queue-based eval system (race condition fix)

## Training the AI

See [training.md](training.md) for the full training pipeline documentation.

```bash
# Supervised pre-training on master games (~30 min on GPU)
python -m training.supervised \
  --pgn data/lichess_elite_2025-01.pgn \
  --max-games 50000 --epochs 15 --export

# Export to ONNX for browser
python -m training.export_onnx --quantize
```

**Model specs**: 6 residual blocks, 128 filters, ~11.4M parameters, AlphaZero 8×8×73 move encoding.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Django 5.2 + Daphne 4.2 (ASGI) |
| WebSocket | Django Channels 4.3 + Redis |
| AI Engine | PyTorch → ONNX Runtime Web (WASM) |
| Position Eval | Stockfish.js v10 |
| LLM Coaching | Groq (Llama 3.3 70B) |
| UI | Bootstrap 5.3 + Chessboard.js 1.0 |
| Chess Logic | Chess.js 0.10 / python-chess 1.11 |

---

## Deployment (Render)

See the [`render` branch](../../tree/render) for Render-specific configuration.

1. Create a Redis instance on Render
2. Set env vars: `REDIS_URL`, `SECRET_KEY`, `GROQ_API_KEY`
3. Build: `sh build.sh` | Start: `daphne pychess.asgi:application --bind 0.0.0.0 -v2`

---

## License

MIT
