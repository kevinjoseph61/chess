"""
Training loop for AlphaZero chess.

Iterates between self-play data generation and neural network training.
Supports checkpointing, evaluation against previous versions, and logging.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
from pathlib import Path

from .model import AlphaZeroNet
from .mcts import MCTS
from .self_play import SelfPlayConfig, TrainingExample, generate_games


class ChessDataset(Dataset):
    """Dataset of (state, policy, value) training examples."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex.state, dtype=torch.float32),
            torch.tensor(ex.policy, dtype=torch.float32),
            torch.tensor(ex.value, dtype=torch.float32),
        )


class TrainingConfig:
    # Self-play
    games_per_iteration: int = 50
    num_simulations: int = 200
    temperature_threshold: int = 30

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 100_000

    # Evaluation
    eval_games: int = 20
    eval_simulations: int = 100
    win_threshold: float = 0.55

    # Infrastructure
    num_iterations: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parallel_games: int = 16

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Current best model
        self.model = AlphaZeroNet().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)

        # Training stats
        self.iteration = 0
        self.total_games = 0

    def run(self):
        """Main training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"{'='*60}")

            # Phase 1: Self-play
            t0 = time.time()
            new_examples = self._self_play()
            self.replay_buffer.extend(new_examples)
            sp_time = time.time() - t0
            print(
                f"Self-play: {len(new_examples)} examples from "
                f"{self.config.games_per_iteration} games ({sp_time:.1f}s)"
            )

            # Phase 2: Training
            t0 = time.time()
            losses = self._train()
            train_time = time.time() - t0
            print(f"Training: avg_loss={np.mean(losses):.4f} ({train_time:.1f}s)")

            # Phase 3: Evaluation (every 5 iterations or final iteration)
            if (iteration + 1) % 5 == 0 or iteration == self.config.num_iterations - 1:
                self._evaluate_and_checkpoint()
            else:
                self._save_checkpoint("latest")

            print(f"Replay buffer size: {len(self.replay_buffer)}")

    def _self_play(self) -> list[TrainingExample]:
        """Generate self-play games using current model."""
        self.model.eval()
        mcts = MCTS(self.model, device=self.device, batch_size=32)
        sp_config = SelfPlayConfig(
            num_simulations=self.config.num_simulations,
            temperature_threshold=self.config.temperature_threshold,
            batch_size=32,
            parallel_games=self.config.parallel_games,
        )

        def progress(game, total, moves):
            print(f"  Game {game}/{total} ({moves} moves)", end="\r")

        examples = generate_games(
            mcts,
            num_games=self.config.games_per_iteration,
            config=sp_config,
            progress_callback=progress,
        )
        print()  # newline after progress
        self.total_games += self.config.games_per_iteration
        return examples

    def _train(self) -> list[float]:
        """Train the network on replay buffer data."""
        self.model.train()

        examples = list(self.replay_buffer)
        random.shuffle(examples)
        dataset = ChessDataset(examples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        losses = []
        for epoch in range(self.config.epochs_per_iteration):
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            batches = 0
            for states, target_policies, target_values in loader:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                # Forward
                policy_logits, values = self.model(states)

                # Policy loss: cross-entropy with MCTS visit distribution
                policy_loss = -torch.sum(
                    target_policies * nn.functional.log_softmax(policy_logits, dim=1)
                ) / states.size(0)

                # Value loss: MSE
                value_loss = nn.functional.mse_loss(values.squeeze(), target_values)

                # Total loss
                loss = policy_loss + value_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            avg_p = epoch_policy_loss / max(batches, 1)
            avg_v = epoch_value_loss / max(batches, 1)
            losses.append(avg_loss)
            print(
                f"  Epoch {epoch+1}/{self.config.epochs_per_iteration}: "
                f"loss={avg_loss:.4f} (policy={avg_p:.4f}, value={avg_v:.4f})",
                flush=True,
            )

        return losses

    def _evaluate_and_checkpoint(self):
        """Evaluate current model against saved best, checkpoint if better."""
        best_path = self.checkpoint_dir / "best.pt"

        if best_path.exists():
            # Load previous best
            prev_model = AlphaZeroNet().to(self.device)
            prev_model.load_state_dict(
                torch.load(best_path, weights_only=True)["model"]
            )

            win_rate = self._play_match(self.model, prev_model)
            print(f"Evaluation: win rate = {win_rate:.1%} vs previous best")

            if win_rate > self.config.win_threshold:
                print("New best model! Saving checkpoint...")
                self._save_checkpoint("best")
            else:
                print("Keeping previous best model.")
        else:
            print("No previous best — saving current as best.")
            self._save_checkpoint("best")

        self._save_checkpoint("latest")

    def _play_match(self, model_a, model_b) -> float:
        """
        Play evaluation games between two models.
        Returns win rate of model_a.
        """
        model_a.eval()
        model_b.eval()

        mcts_a = MCTS(model_a, device=self.device, batch_size=32)
        mcts_b = MCTS(model_b, device=self.device, batch_size=32)

        wins = 0
        draws = 0
        num_games = self.config.eval_games

        for game_idx in range(num_games):
            # Alternate colors
            if game_idx % 2 == 0:
                white_mcts, black_mcts = mcts_a, mcts_b
                a_is_white = True
            else:
                white_mcts, black_mcts = mcts_b, mcts_a
                a_is_white = False

            result = self._play_eval_game(white_mcts, black_mcts)

            if result == 0.5:
                draws += 1
            elif (result == 1.0 and a_is_white) or (result == 0.0 and not a_is_white):
                wins += 1

        return (wins + 0.5 * draws) / num_games

    def _play_eval_game(self, white_mcts, black_mcts) -> float:
        """Play a single evaluation game. Returns 1.0/0.0/0.5 for white win/loss/draw."""
        import chess

        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < 150:
            mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            move, _ = mcts.search(
                board,
                num_simulations=self.config.eval_simulations,
                temperature=0.1,
                add_noise=False,
            )
            board.push(move)
            move_count += 1

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return 0.0
        return 0.5

    def _save_checkpoint(self, name: str):
        """Save a model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "total_games": self.total_games,
            },
            path,
        )

    def load_checkpoint(self, name: str = "latest"):
        """Load a model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False
        checkpoint = torch.load(path, weights_only=True)
        # Support both checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.total_games = checkpoint.get("total_games", 0)
        print(f"Loaded checkpoint '{name}' (iteration {self.iteration})")
        return True

    def load_pretrained(self, path: str):
        """Load model weights from any checkpoint (e.g., supervised pre-training).
        Only loads model weights, resets optimizer and iteration count."""
        checkpoint = torch.load(path, weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        # Reset optimizer for fresh fine-tuning
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.iteration = 0
        self.total_games = 0
        print(f"Loaded pretrained weights from {path}")
