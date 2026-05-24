# Chess Gang — System Architecture

## Overview

A real-time chess platform with AI coaching, built on Django/Channels with a browser-side neural network engine.

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Client)                     │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Chess UI │  │ Engine Worker │  │ Stockfish Worker │  │
│  │ (Chess.js│  │ (MCTS + ONNX │  │ (UCI eval,       │  │
│  │  Board.js│  │  Runtime Web)│  │  depth 12)       │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘  │
│       │               │                    │            │
│       └───────┬───────┘────────────────────┘            │
│               │                                         │
│         ┌─────┴──────┐                                  │
│         │ Coaching UI │ ← LLM tips, move badges, eval  │
│         └─────┬──────┘                                  │
└───────────────┼─────────────────────────────────────────┘
                │ HTTP (POST /api/coach/, /api/analyze/)
                │ WebSocket (multiplayer only)
┌───────────────┼─────────────────────────────────────────┐
│               │          Server (Django/Daphne)          │
│         ┌─────┴──────┐                                  │
│         │   Views     │ → coach_move_api, analyze        │
│         └─────┬──────┘                                  │
│               │                                         │
│         ┌─────┴──────┐  ┌───────────┐  ┌────────────┐  │
│         │ analysis.py │  │ Channels  │  │  Models    │  │
│         │ (Groq LLM) │  │ WebSocket │  │ (Game, etc)│  │
│         └─────┬──────┘  └─────┬─────┘  └─────┬──────┘  │
│               │               │               │         │
│         ┌─────┴──┐     ┌─────┴─────┐   ┌────┴─────┐   │
│         │  Groq  │     │   Redis   │   │ SQLite   │   │
│         │  API   │     │ (pub/sub) │   │ (DB)     │   │
│         └────────┘     └───────────┘   └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Frontend — Single Player (`game/templates/game/single.html`)

The single-player page runs two Web Workers in parallel:

| Worker | File | Purpose |
|--------|------|---------|
| **Engine Worker** | `engine-worker.js` | Neural net AI (MCTS + ONNX model) |
| **Stockfish Worker** | `stockfish.js` | Position evaluation for move classification |

**Chess UI**: Chessboard.js 1.0.0 (rendering) + Chess.js 0.10.2 (move validation/PGN).

**Coaching Panel**: Right sidebar showing per-move badges (Brilliant/Good/OK/Inaccuracy/Mistake/Blunder), centipawn loss, eval bar, score tracker, and LLM coaching tips.

### 2. Frontend — Multiplayer (`game/templates/game/game.html`)

Uses WebSocket (via ReconnectingWebSocket) to Django Channels for real-time move synchronization. No AI engine — both players are human.

### 3. Backend — Django + Daphne (ASGI)

| File | Responsibility |
|------|---------------|
| `pychess/routing.py` | ASGI routing — HTTP via `get_asgi_application()`, WebSocket via Channels |
| `pychess/urls.py` | URL routes — lobby, game, single, create, ongoing, completed, API endpoints |
| `game/views.py` | View logic — game creation, user registration, coaching API |
| `game/consumers.py` | WebSocket consumers — `GameConsumer` (multiplayer), `SingleConsumer` (legacy) |
| `game/analysis.py` | LLM integration — `analyze_game()` (post-game), `coach_move()` (per-move) |
| `game/models.py` | Database models — `Game` (status, PGN, FEN, players, online status) |

### 4. AI Training Pipeline (`training/`)

Offline pipeline for training the neural network model:

```
training/
├── model.py          # AlphaZeroNet (6 residual blocks, 128 filters, ~11.4M params)
├── encoding.py       # Board ↔ tensor, move ↔ index (AlphaZero 8×8×73 scheme)
├── mcts.py           # Monte Carlo Tree Search with batched GPU evaluation
├── self_play.py      # Self-play game generation with resignation
├── train.py          # Self-play RL training loop
├── supervised.py     # Supervised pre-training on master games
├── run_training.py   # CLI entry point
├── export_onnx.py    # PyTorch → ONNX export + INT8 quantization
```

### 5. LLM Coaching (Groq API)

Uses **Llama 3.3 70B** via Groq's free tier for:
- **Per-move coaching**: 1-2 sentence tips after each player move (~0.5s response time)
- **Post-game analysis**: Full game review with critical moments, missed tactics, strategic themes

---

## Data Flow — Single Player Move

```
1. Player drops piece on board
   ├─→ Chess.js validates move, updates PGN
   ├─→ PGN snapshot captured (for coaching)
   ├─→ Stockfish Worker: cancel baseline search → eval post-move position
   └─→ Engine Worker: search for AI's response

2. Stockfish completes post-move eval (~1-2s at depth 12)
   ├─→ Compute cpLoss = baseline - (-eval_after)
   ├─→ Classify: Brilliant/Good/OK/Inaccuracy/Mistake/Blunder
   ├─→ Show badge + eval bar update
   └─→ POST /api/coach/ with PGN snapshot + category

3. Engine Worker finds AI move (~1-5s depending on level)
   ├─→ Apply AI move to board
   └─→ Stockfish Worker: eval new position (baseline for next move)

4. Groq LLM returns coaching tip (~0.5s)
   └─→ Display in coaching panel
```

---

## Database Schema

**Game model** (`game/models.py`):
- `owner` / `opponent`: ForeignKey to User
- `owner_side`: 'white' or 'black'
- `status`: 1=awaiting, 2=in_progress, 3=completed
- `pgn` / `fen`: Game state
- `winner`: Result string
- `owner_online` / `opponent_online`: Boolean (WebSocket presence)
- `level`: Difficulty for public games

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Server | Django + Daphne (ASGI) | 5.2.14 / 4.2.1 |
| WebSocket | Django Channels + Redis | 4.3.2 |
| Database | SQLite | 3.x |
| AI Model | PyTorch → ONNX | 2.4.1 / 1.21.0 |
| Browser AI | ONNX Runtime Web (WASM) | 1.21.0 |
| Position Eval | Stockfish.js (v10) | 10.0.2 |
| LLM | Groq (Llama 3.3 70B) | Free tier |
| UI | Bootstrap 5.3.8 + Chessboard.js | Dark theme |
| Chess Logic | Chess.js / python-chess | 0.10.2 / 1.11.2 |
