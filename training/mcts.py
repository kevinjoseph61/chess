"""
Monte Carlo Tree Search (MCTS) for AlphaZero chess.

Uses the neural network to guide search:
- Policy network provides prior probabilities for move selection (exploration)
- Value network evaluates leaf nodes (instead of random rollouts)

Optimized with:
- Batched leaf evaluation (sends multiple positions to GPU at once)
- Virtual losses to enable parallel tree traversal
"""

import math
import chess
import numpy as np
import torch

from .encoding import board_to_tensor, move_to_index, index_to_move, get_legal_move_mask

# MCTS hyperparameters
C_PUCT = 1.5  # Exploration constant
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25  # Fraction of Dirichlet noise added to root priors
VIRTUAL_LOSS = 3  # Virtual loss for parallel traversal


class MCTSNode:
    __slots__ = [
        "parent",
        "move",
        "prior",
        "children",
        "visit_count",
        "total_value",
        "board",
    ]

    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits):
        """Upper confidence bound for tree search."""
        exploration = (
            C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )
        return self.q_value + exploration

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        """Select the child with the highest UCB score."""
        return max(self.children, key=lambda c: c.ucb_score(self.visit_count))

    def expand(self, policy_priors):
        """
        Expand this node by creating children for all legal moves.
        policy_priors: dict mapping move_index -> prior probability
        """
        for move in self.board.legal_moves:
            flip = not self.board.turn
            idx = move_to_index(move, flip)
            prior = policy_priors.get(idx, 1e-8)
            child_board = self.board.copy()
            child_board.push(move)
            child = MCTSNode(child_board, parent=self, move=move, prior=prior)
            self.children.append(child)

    def backpropagate(self, value):
        """Propagate value up the tree. Value is negated at each level."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # opponent's perspective
            node = node.parent


class MCTS:
    def __init__(self, model, device="cpu", batch_size=8):
        self.model = model
        self.device = device
        self.batch_size = batch_size  # Number of leaves to evaluate per GPU batch
        self.root = None  # Last search root, for accessing value

    @torch.no_grad()
    def _evaluate_batch(self, boards):
        """
        Evaluate multiple board positions in a single GPU batch.
        Returns list of (policy_dict, value) tuples.
        """
        tensors = [board_to_tensor(b) for b in boards]
        x = torch.tensor(np.stack(tensors), dtype=torch.float32).to(self.device)

        policy_logits_batch, values_batch = self.model(x)
        policy_logits_batch = policy_logits_batch.cpu().numpy()
        values_batch = values_batch.cpu().numpy()

        results = []
        for i, board in enumerate(boards):
            policy_logits = policy_logits_batch[i]
            value = float(values_batch[i])

            mask = get_legal_move_mask(board)
            policy_logits[mask == 0] = -1e9
            policy = _softmax(policy_logits)

            policy_dict = {}
            flip = not board.turn
            for move in board.legal_moves:
                idx = move_to_index(move, flip)
                policy_dict[idx] = policy[idx]

            results.append((policy_dict, value))

        return results

    @torch.no_grad()
    def _evaluate(self, board):
        """Single board evaluation (fallback)."""
        results = self._evaluate_batch([board])
        return results[0]

    def search(self, board, num_simulations, temperature=1.0, add_noise=True):
        """
        Run MCTS from the given position using batched leaf evaluation.

        Collects multiple leaf nodes per batch before sending to GPU,
        dramatically improving GPU utilization.
        """
        root = MCTSNode(board.copy())
        self.root = root  # store for external access

        # Evaluate root and expand
        policy_priors, _ = self._evaluate(board)
        root.expand(policy_priors)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - DIRICHLET_FRAC) * child.prior + DIRICHLET_FRAC * n

        # Run simulations in batches
        sims_done = 0
        while sims_done < num_simulations:
            # Collect a batch of leaf nodes using virtual losses
            batch_size = min(self.batch_size, num_simulations - sims_done)
            leaves = []
            leaf_paths = []

            for _ in range(batch_size):
                node = root

                # Selection with virtual loss
                while not node.is_leaf():
                    node = node.select_child()

                # Check terminal state
                if node.board.is_game_over():
                    result = node.board.result()
                    if result == "1-0":
                        value = 1.0 if node.board.turn == chess.BLACK else -1.0
                    elif result == "0-1":
                        value = 1.0 if node.board.turn == chess.WHITE else -1.0
                    else:
                        value = 0.0
                    node.backpropagate(-value)
                    sims_done += 1
                    continue

                # Apply virtual loss to discourage other paths from picking same node
                node.visit_count += VIRTUAL_LOSS
                node.total_value -= VIRTUAL_LOSS

                leaves.append(node)

            if not leaves:
                continue

            # Batch evaluate all collected leaves
            boards = [leaf.board for leaf in leaves]
            results = self._evaluate_batch(boards)

            # Expand and backpropagate each leaf
            for leaf, (policy_priors, value) in zip(leaves, results):
                # Undo virtual loss
                leaf.visit_count -= VIRTUAL_LOSS
                leaf.total_value += VIRTUAL_LOSS

                leaf.expand(policy_priors)
                leaf.backpropagate(-value)
                sims_done += 1

        # Extract visit counts for policy target
        flip = not board.turn
        policy_target = np.zeros(4672, dtype=np.float32)

        for child in root.children:
            idx = move_to_index(child.move, flip)
            policy_target[idx] = child.visit_count

        # Normalize policy target
        total = policy_target.sum()
        if total > 0:
            policy_target /= total

        # Select move based on temperature
        if temperature < 0.01:
            best_child = max(root.children, key=lambda c: c.visit_count)
            best_move = best_child.move
        else:
            visits = np.array([c.visit_count for c in root.children], dtype=np.float64)
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            chosen_idx = np.random.choice(len(root.children), p=probs)
            best_move = root.children[chosen_idx].move

        return best_move, policy_target


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
