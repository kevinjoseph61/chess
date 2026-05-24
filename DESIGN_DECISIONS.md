# Design Decisions

Key architectural and design decisions made during development, with rationale.

---

## 1. Browser-Side AI Engine (vs Server-Side)

**Decision**: Run the chess AI entirely in the browser via Web Workers + ONNX Runtime Web.

**Alternatives considered**:
- Server-side inference (Django view)
- Cloud GPU inference (API call)

**Rationale**:
- **Zero server compute cost** — critical for free-tier hosting on Render
- **No latency** — no round-trip to server for each AI move
- **Scales to unlimited users** — each browser runs its own engine
- **Offline capable** — works without internet after initial page load (except coaching)
- **Trade-off**: Model must be small enough for browser (~11 MB quantized ONNX)

---

## 2. Dual Worker Architecture (Neural Net + Stockfish)

**Decision**: Use two separate Web Workers — one for AI move generation (neural net MCTS), one for position evaluation (Stockfish).

**Why not use Stockfish for everything?**
- The neural net engine provides the interesting AI experience (AlphaZero-style play)
- Stockfish is too strong at any meaningful depth — not fun for beginners
- The neural net's difficulty scales naturally with simulation count

**Why not use the neural net for evaluation?**
- The value head gives relative assessments but isn't calibrated in centipawns
- Stockfish provides precise, industry-standard centipawn evaluations needed for move classification
- Stockfish at depth 12 gives reasonable evals in ~1-2 seconds

**Race condition handling**: The two workers run asynchronously. A queue-based system (`sfEvalQueue`) ensures correct ordering of baseline vs post-move evaluations. When the player moves before the baseline eval completes, the in-progress search is cancelled via UCI `stop`.

---

## 3. Supervised Pre-Training (vs Pure Self-Play)

**Decision**: Train the neural net primarily via supervised learning on master games, with optional self-play fine-tuning.

**Rationale**:
- Pure AlphaZero self-play requires ~5,000 TPUs for 9 hours (Google's setup)
- On a single RTX 3060, self-play produces only draws for hundreds of iterations
- 50k master games + 15 epochs gives 37% move accuracy in ~30 minutes
- The model immediately plays recognizable openings and understands piece values
- Self-play fine-tuning can improve tactical sharpness after the supervised foundation

---

## 4. ONNX Quantization (INT8)

**Decision**: Export the model as INT8 quantized ONNX (10.9 MB) rather than full-precision (43 MB).

**Rationale**:
- 4× smaller file = faster page load over mobile networks
- Minimal accuracy loss for chess inference
- ONNX Runtime Web handles INT8 dequantization transparently
- Both `model.onnx` (43 MB) and `model_q8.onnx` (10.9 MB) are available; the browser uses the quantized version

---

## 5. Groq for LLM Coaching (vs OpenAI/Anthropic)

**Decision**: Use Groq's free tier with Llama 3.3 70B for coaching.

**Rationale**:
- **Free tier** — no cost for the coaching feature
- **Fast inference** — Groq's LPU gives ~0.5s response times (vs 2-3s for OpenAI)
- **Llama 3.3 70B** — strong enough for chess analysis, available on free tier
- **Trade-off**: Rate limits on free tier (30 RPM), but sufficient for per-move coaching

---

## 6. Per-Move Coaching (vs Post-Game Only)

**Decision**: Provide real-time coaching on every move, not just post-game analysis.

**Rationale**:
- Immediate feedback is more pedagogically effective
- Players can adjust their play based on coaching tips
- Post-game analysis is also available for deeper review
- Uses PGN snapshot captured before AI responds, ensuring coaching always references the player's move

**Implementation detail**: The PGN is captured at `onDrop()` time (before the AI responds), stored as `sfPendingPGN`, and sent to the coaching API. This prevents the LLM from accidentally coaching the AI's response move.

---

## 7. Centipawn Loss Thresholds

**Decision**: Use calibrated thresholds for move classification based on Stockfish v10 depth-12 evals.

| Category | cpLoss (pawns) | Score Delta |
|----------|---------------|-------------|
| Brilliant | ≤ -1.0 | +3 |
| Good | ≤ 0.5 | +1 |
| OK | ≤ 1.0 | 0 |
| Inaccuracy | ≤ 2.0 | -1 |
| Mistake | ≤ 4.0 | -2 |
| Blunder | > 4.0 | -3 |

**Why these thresholds?** Stockfish v10 at depth 12 gives noisier evals than modern Stockfish 16+. Wider thresholds (especially Good ≤ 0.5 pawns) prevent normal opening moves from being flagged as inaccuracies.

**Mate score handling**: Mate scores (±100 pawns) are clamped to avoid absurd centipawn values. Finding a forced mate = Brilliant (-1.5 cpLoss), missing/allowing mate = Blunder (5.0 cpLoss).

---

## 8. Bootstrap 5 Dark Theme

**Decision**: Custom dark theme using CSS variables across all 9 templates.

**Rationale**:
- Dark themes reduce eye strain during extended play sessions
- Chess boards look better on dark backgrounds (less visual noise)
- CSS variables (`--chess-bg`, `--chess-surface`, `--chess-highlight`) enable easy theme customization
- All templates share `layout.html` base with consistent navbar, icons, and styling

---

## 9. Django Channels + Redis (Multiplayer)

**Decision**: Use Django Channels with Redis channel layer for real-time multiplayer.

**Rationale**:
- WebSocket support required for real-time move synchronization
- Django Channels integrates naturally with Django's auth and ORM
- Redis provides pub/sub for multi-server deployments (Render)
- ReconnectingWebSocket on the client handles network interruptions

**Status tracking**: `owner_online`/`opponent_online` booleans in the Game model, updated on WebSocket connect/disconnect. Players see "Waiting for opponent" or "Opponent disconnected" modals.

---

## 10. Stockfish.js Loaded Directly (vs Wrapper)

**Decision**: Load `stockfish.js` directly as a Web Worker, not through a wrapper script.

**Why**: Stockfish.js v10.0.2 auto-initializes when loaded as a Worker. The `STOCKFISH()` constructor pattern from newer versions doesn't apply. A wrapper using `importScripts()` fails with cross-origin restrictions when loaded from CDN, so the file is hosted locally in `game/static/engine/`.

---

## 11. Queue-Based Eval System (vs Boolean Flag)

**Decision**: Replace `sfPendingPlayerMove` boolean with `sfEvalQueue` array.

**Problem**: The boolean flag created a race condition — if the player moved before the baseline eval completed, the baseline result was misinterpreted as the post-move result, causing incorrect classifications (e.g., Kd7 classified as "Brilliant").

**Solution**: An array queue (`['baseline']` or `['postmove']`) tracks the expected result type. On player move:
1. Send `stop` to cancel in-progress baseline search
2. Clear the queue (`sfEvalQueue = []`)
3. Push `'postmove'` and start new eval

The cancelled baseline's `bestmove` response finds an empty queue, falls to the `else` branch harmlessly, and the subsequent postmove result is correctly classified.
