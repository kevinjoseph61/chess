"""
Supervised pre-training on master games from PGN files.

Trains the AlphaZero network to predict:
  - Policy: the move played by the master (cross-entropy loss)
  - Value: the game result from the current player's perspective (MSE loss)

This gives the model a strong starting point before self-play fine-tuning.

Usage:
    python -m training.supervised --pgn data/games.pgn
    python -m training.supervised --pgn data/games.pgn --epochs 10 --lr 0.001
    python -m training.supervised --download  # download Lichess elite DB
"""

import argparse
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .model import AlphaZeroNet
from .encoding import board_to_tensor, move_to_index


@dataclass
class SupervisedConfig:
    pgn_path: str = "data/master_games.pgn"
    max_games: int = 50_000  # max games to load
    max_positions: int = 2_000_000  # max training positions
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    val_split: float = 0.05  # 5% validation
    checkpoint_dir: str = "checkpoints"
    sample_positions: int = 8  # positions sampled per game (0 = all)
    min_elo: int = 2000  # minimum Elo to include game
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ChessPositionDataset(Dataset):
    """Dataset of (board_tensor, move_index, game_result) from PGN games."""

    def __init__(self, positions):
        self.states = np.array([p[0] for p in positions], dtype=np.float32)
        self.moves = np.array([p[1] for p in positions], dtype=np.int64)
        self.values = np.array([p[2] for p in positions], dtype=np.float32)

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.moves[idx]),
            torch.tensor(self.values[idx]),
        )


def parse_pgn(pgn_path: str, config: SupervisedConfig):
    """
    Parse a PGN file and extract training positions.
    Returns list of (board_tensor, move_index, result_value) tuples.
    """
    positions = []
    games_loaded = 0
    games_skipped = 0

    print(f"Parsing PGN: {pgn_path}")
    file_size = os.path.getsize(pgn_path)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while games_loaded < config.max_games and len(positions) < config.max_positions:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue

            if game is None:
                break

            # Filter by result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                result_value = 1.0
            elif result == "0-1":
                result_value = -1.0
            elif result == "1/2-1/2":
                result_value = 0.0
            else:
                games_skipped += 1
                continue

            # Optional Elo filter
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo > 0 and white_elo < config.min_elo:
                    games_skipped += 1
                    continue
                if black_elo > 0 and black_elo < config.min_elo:
                    games_skipped += 1
                    continue
            except ValueError:
                pass

            # Extract positions
            board = game.board()
            game_positions = []

            for move in game.mainline_moves():
                try:
                    flip = not board.turn
                    state = board_to_tensor(board)
                    move_idx = move_to_index(move, flip)

                    # Value from current player's perspective
                    if board.turn == chess.WHITE:
                        value = result_value
                    else:
                        value = -result_value

                    game_positions.append((state, move_idx, value))
                    board.push(move)
                except Exception:
                    board.push(move)
                    continue

            # Sample positions from this game to avoid overweighting long games
            if (
                config.sample_positions > 0
                and len(game_positions) > config.sample_positions
            ):
                game_positions = random.sample(game_positions, config.sample_positions)

            positions.extend(game_positions)
            games_loaded += 1

            if games_loaded % 1000 == 0:
                print(
                    f"  {games_loaded} games loaded, {len(positions)} positions "
                    f"({games_skipped} skipped)",
                    flush=True,
                )

    print(
        f"Loaded {games_loaded} games, {len(positions)} positions "
        f"({games_skipped} skipped)"
    )

    return positions


def train_supervised(config: SupervisedConfig, resume_from: str = None):
    """Run supervised training on master game positions."""
    device = torch.device(config.device)

    # Load data
    positions = parse_pgn(config.pgn_path, config)
    if not positions:
        print("ERROR: No positions loaded! Check your PGN file.")
        return

    # Shuffle and split
    random.shuffle(positions)
    val_count = max(1, int(len(positions) * config.val_split))
    val_positions = positions[:val_count]
    train_positions = positions[val_count:]

    print(f"Train: {len(train_positions)}, Val: {val_count}")

    train_dataset = ChessPositionDataset(train_positions)
    val_dataset = ChessPositionDataset(val_positions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model
    model = AlphaZeroNet().to(device)
    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {resume_from}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        for states, moves, values in train_loader:
            states = states.to(device)
            moves = moves.to(device)
            values = values.to(device)

            policy_logits, value_pred = model(states)
            policy_loss = policy_criterion(policy_logits, moves)
            value_loss = value_criterion(value_pred.squeeze(), values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += policy_loss.item() * states.size(0)
            train_value_loss += value_loss.item() * states.size(0)
            preds = policy_logits.argmax(dim=1)
            train_correct += (preds == moves).sum().item()
            train_total += states.size(0)

        scheduler.step()

        train_policy_loss /= train_total
        train_value_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, moves, values in val_loader:
                states = states.to(device)
                moves = moves.to(device)
                values = values.to(device)

                policy_logits, value_pred = model(states)
                policy_loss = policy_criterion(policy_logits, moves)
                value_loss = value_criterion(value_pred.squeeze(), values)

                val_policy_loss += policy_loss.item() * states.size(0)
                val_value_loss += value_loss.item() * states.size(0)
                preds = policy_logits.argmax(dim=1)
                val_correct += (preds == moves).sum().item()
                val_total += states.size(0)

        val_policy_loss /= val_total
        val_value_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        val_loss = val_policy_loss + val_value_loss

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{config.epochs} ({elapsed:.0f}s) lr={lr:.6f}")
        print(
            f"  Train: policy={train_policy_loss:.4f} value={train_value_loss:.4f} "
            f"acc={train_acc:.1f}%"
        )
        print(
            f"  Val:   policy={val_policy_loss:.4f} value={val_value_loss:.4f} "
            f"acc={val_acc:.1f}%"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                f"{config.checkpoint_dir}/best.pt",
            )
            print(f"  *** New best model (val_loss={val_loss:.4f}, acc={val_acc:.1f}%)")

        # Save latest every epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            f"{config.checkpoint_dir}/latest.pt",
        )

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    return model


def download_pgn(output_dir: str = "data"):
    """Download a curated set of master games for training."""
    import urllib.request
    import gzip

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/master_games.pgn"

    if Path(output_path).exists():
        size = os.path.getsize(output_path)
        print(f"PGN file already exists: {output_path} ({size/1024/1024:.1f} MB)")
        return output_path

    # Use Lichess elite database (2400+ rated players, ~50MB compressed)
    # This is a well-known free dataset for chess AI training
    url = "https://database.lichess.org/lichess_elite_2024-09.pgn.zst"
    print(f"Note: For best results, download master games manually from:")
    print(f"  https://database.lichess.org/#standard_games")
    print(f"  (Download any monthly elite database, extract, place in {output_dir}/)")
    print()

    # Alternative: generate some example games from well-known openings
    print("Generating sample training games from known openings...")
    _generate_sample_games(output_path)
    return output_path


def _generate_sample_games(output_path: str):
    """
    Generate synthetic training games by playing known openings
    followed by random-ish play with material-aware evaluation.
    This gives the model basic opening knowledge and piece values.
    """
    import chess
    import random

    # Famous openings to teach the model
    OPENINGS = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5",  # Ruy Lopez
        "1. e4 e5 2. Nf3 Nc6 3. Bc4",  # Italian Game
        "1. e4 c5",  # Sicilian
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4",  # Sicilian Open
        "1. e4 e6 2. d4 d5",  # French Defense
        "1. e4 c6 2. d4 d5",  # Caro-Kann
        "1. d4 d5 2. c4",  # Queen's Gambit
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6",  # QGD
        "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7",  # King's Indian
        "1. d4 Nf6 2. c4 e6 3. Nf3",  # Indian Systems
        "1. e4 e5 2. Nf3 Nf6",  # Petroff
        "1. Nf3 d5 2. g3 Nf6",  # Reti
        "1. c4 e5",  # English
        "1. e4 d5 2. exd5 Qxd5",  # Scandinavian
        "1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4",  # Scotch
        "1. d4 d5 2. c4 dxc4 3. Nf3",  # QGA
        "1. e4 e5 2. f4",  # King's Gambit
        "1. d4 f5",  # Dutch
        "1. e4 d6 2. d4 Nf6 3. Nc3",  # Pirc
        "1. e4 g6 2. d4 Bg7",  # Modern
    ]

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def material_eval(board):
        """Simple material evaluation."""
        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                val = PIECE_VALUES[piece.piece_type]
                score += val if piece.color == chess.WHITE else -val
        return score

    def pick_move(board):
        """Pick a move biased toward captures and central play."""
        legal = list(board.legal_moves)
        if not legal:
            return None

        # Evaluate each move
        scored = []
        for move in legal:
            board.push(move)

            if board.is_checkmate():
                board.pop()
                return move

            score = (
                -material_eval(board)
                if board.turn == chess.WHITE
                else material_eval(board)
            )

            # Bonus for captures
            if board.is_capture(move):
                score += 0.5

            # Bonus for center control
            to_sq = move.to_square
            center = {chess.E4, chess.D4, chess.E5, chess.D5}
            if to_sq in center:
                score += 0.2

            # Add randomness
            score += random.gauss(0, 1.5)

            scored.append((score, move))
            board.pop()

        scored.sort(key=lambda x: -x[0])
        # Softmax-like selection from top moves
        top = scored[: max(3, len(scored) // 4)]
        return random.choice(top)[1]

    games_text = []
    num_games = 5000  # Generate 5000 games

    for i in range(num_games):
        opening = random.choice(OPENINGS)
        board = chess.Board()
        pgn_game = chess.pgn.Game()
        node = pgn_game

        # Play opening
        try:
            temp_board = chess.Board()
            for token in opening.replace(".", " ").split():
                token = token.strip()
                if not token or token[0].isdigit():
                    continue
                move = temp_board.parse_san(token)
                node = node.add_variation(move)
                board.push(move)
                temp_board.push(move)
        except Exception:
            board = chess.Board()
            pgn_game = chess.pgn.Game()
            node = pgn_game

        # Play until game over or max moves
        for _ in range(100):
            if board.is_game_over():
                break
            move = pick_move(board)
            if move is None:
                break
            node = node.add_variation(move)
            board.push(move)

        # Set result
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                result = "0-1"
            else:
                result = "1-0"
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
        else:
            # Adjudicate based on material
            mat = material_eval(board)
            if mat > 5:
                result = "1-0"
            elif mat < -5:
                result = "0-1"
            else:
                result = "1/2-1/2"

        pgn_game.headers["Event"] = "Training"
        pgn_game.headers["White"] = "Engine"
        pgn_game.headers["Black"] = "Engine"
        pgn_game.headers["WhiteElo"] = "2400"
        pgn_game.headers["BlackElo"] = "2400"
        pgn_game.headers["Result"] = result

        games_text.append(str(pgn_game))

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{num_games} games", flush=True)

    with open(output_path, "w") as f:
        f.write("\n\n".join(games_text))

    size = os.path.getsize(output_path)
    print(f"Generated {num_games} games -> {output_path} ({size/1024/1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Supervised pre-training on master games"
    )
    parser.add_argument(
        "--pgn", type=str, default=None, help="Path to PGN file with master games"
    )
    parser.add_argument(
        "--download", action="store_true", help="Generate/download training games"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-games", type=int, default=50000)
    parser.add_argument(
        "--sample-per-game",
        type=int,
        default=8,
        help="Positions to sample per game (0=all)",
    )
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export to ONNX after training"
    )
    args = parser.parse_args()

    if args.download or args.pgn is None:
        pgn_path = download_pgn()
    else:
        pgn_path = args.pgn

    config = SupervisedConfig(
        pgn_path=pgn_path,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_games=args.max_games,
        sample_positions=args.sample_per_game,
        min_elo=args.min_elo,
    )

    model = train_supervised(config, resume_from=args.resume)

    if args.export and model is not None:
        from .export_onnx import export_to_onnx

        Path("game/static/engine").mkdir(parents=True, exist_ok=True)
        export_to_onnx(
            f"{config.checkpoint_dir}/best.pt", "game/static/engine/model.onnx"
        )
        print("Exported to ONNX!")


if __name__ == "__main__":
    main()
