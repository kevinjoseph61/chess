"""
Self-play game generation for AlphaZero training.

Plays games using the current neural network + MCTS, collecting
(board_state, policy_target, game_result) training examples.

Uses batched GPU evaluation across multiple simultaneous games.
"""

import time
import chess
import numpy as np
import torch
from dataclasses import dataclass, field

from .encoding import board_to_tensor, move_to_index, get_legal_move_mask
from .mcts import MCTS


@dataclass
class TrainingExample:
    state: np.ndarray  # (18, 8, 8) board tensor
    policy: np.ndarray  # (4672,) MCTS visit count distribution
    value: float = 0.0  # game outcome from this player's perspective


@dataclass
class SelfPlayConfig:
    num_simulations: int = 200  # MCTS simulations per move
    temperature_threshold: int = 30  # use temperature=1 for first N moves, then 0.1
    max_moves: int = 120  # max moves per game before declaring draw
    batch_size: int = 32  # GPU batch size for MCTS leaf evaluation
    parallel_games: int = 16  # Number of simultaneous games in batch
    resign_threshold: float = -0.9  # resign if value below this for N consecutive moves
    resign_check_moves: int = 5  # how many consecutive moves below threshold to resign


def play_game(
    mcts: MCTS, config: SelfPlayConfig, game_num: int = 0, total_games: int = 0
) -> list[TrainingExample]:
    """
    Play a single self-play game, returning training examples.
    The value field is filled in after the game ends with the actual result.
    """
    board = chess.Board()
    examples = []
    move_count = 0
    game_start = time.time()

    while not board.is_game_over() and move_count < config.max_moves:
        state = board_to_tensor(board)
        temp = 1.0 if move_count < config.temperature_threshold else 0.1

        move, policy_target = mcts.search(
            board,
            num_simulations=config.num_simulations,
            temperature=temp,
            add_noise=True,
        )

        examples.append(TrainingExample(state=state, policy=policy_target, value=0.0))
        board.push(move)
        move_count += 1

    # Determine game result
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
    else:
        final_value = 0.0

    # Fill in values from each player's perspective
    for i, ex in enumerate(examples):
        if i % 2 == 0:
            ex.value = final_value
        else:
            ex.value = -final_value

    return examples


class ParallelGameState:
    """Tracks the state of one game in a parallel batch."""

    def __init__(self, game_id):
        self.game_id = game_id
        self.board = chess.Board()
        self.examples = []
        self.move_count = 0
        self.finished = False
        self.result = 0.0
        self.consecutive_low_evals = 0  # track resignation

    def is_done(self, max_moves):
        return (
            self.finished or self.board.is_game_over() or self.move_count >= max_moves
        )

    def finalize(self):
        """Determine game result and fill in values."""
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                self.result = 1.0
            elif result == "0-1":
                self.result = -1.0
            else:
                self.result = 0.0
        else:
            self.result = 0.0

        for i, ex in enumerate(self.examples):
            if i % 2 == 0:
                ex.value = self.result
            else:
                ex.value = -self.result

        self.finished = True


def generate_games(
    mcts: MCTS, num_games: int, config: SelfPlayConfig, progress_callback=None
) -> list[TrainingExample]:
    """
    Generate self-play games with batched GPU evaluation.
    Multiple games advance simultaneously, sharing MCTS batch evaluations.
    """
    all_examples = []
    games_completed = 0
    total_start = time.time()
    parallel = min(config.parallel_games, num_games)

    print(
        f"  Starting {num_games} games ({parallel} in parallel, "
        f"{config.num_simulations} sims/move)",
        flush=True,
    )

    while games_completed < num_games:
        batch_count = min(parallel, num_games - games_completed)
        games = [ParallelGameState(i) for i in range(batch_count)]
        active_games = games.copy()
        batch_moves = 0

        while active_games:
            for game in active_games:
                state = board_to_tensor(game.board)
                temp = 1.0 if game.move_count < config.temperature_threshold else 0.1

                move, policy_target = mcts.search(
                    game.board,
                    num_simulations=config.num_simulations,
                    temperature=temp,
                    add_noise=True,
                )

                # Check resignation
                root_value = mcts.root.q_value
                if root_value < config.resign_threshold:
                    game.consecutive_low_evals += 1
                else:
                    game.consecutive_low_evals = 0

                if (
                    game.consecutive_low_evals >= config.resign_check_moves
                    and game.move_count > 20
                ):
                    if game.board.turn == chess.WHITE:
                        game.result = -1.0
                    else:
                        game.result = 1.0
                    game.finished = True
                    for i, ex in enumerate(game.examples):
                        ex.value = game.result if i % 2 == 0 else -game.result
                    continue

                game.examples.append(
                    TrainingExample(state=state, policy=policy_target, value=0.0)
                )
                game.board.push(move)
                game.move_count += 1

            batch_moves += 1

            newly_finished = [
                g for g in active_games if g.is_done(config.max_moves) or g.finished
            ]
            for g in newly_finished:
                if not g.finished:
                    g.finalize()
                all_examples.extend(g.examples)
                games_completed += 1

                result_str = {1.0: "White wins", -1.0: "Black wins"}.get(
                    g.result, "Draw"
                )
                elapsed = time.time() - total_start
                print(
                    f"  [{games_completed}/{num_games}] {result_str} "
                    f"in {g.move_count} moves | Total: {elapsed:.0f}s",
                    flush=True,
                )

            active_games = [
                g
                for g in active_games
                if not g.finished and not g.is_done(config.max_moves)
            ]

            if batch_moves % 10 == 0 and active_games:
                elapsed = time.time() - total_start
                avg_moves = sum(g.move_count for g in active_games) / len(active_games)
                print(
                    f"    {len(active_games)} games active, "
                    f"avg {avg_moves:.0f} moves | {elapsed:.0f}s",
                    flush=True,
                )

    total_elapsed = time.time() - total_start
    print(
        f"  Self-play complete: {num_games} games, "
        f"{len(all_examples)} examples, {total_elapsed:.1f}s",
        flush=True,
    )

    return all_examples
