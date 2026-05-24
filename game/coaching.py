"""
Real-time move coaching for single player mode.

Evaluates the player's move by comparing position evaluation before
and after the move. Returns a coaching message with a quality rating.
"""

import chess
import chess.pgn
import io

# Piece-square tables for positional evaluation (centipawns)
PAWN_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

KNIGHT_TABLE = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]

BISHOP_TABLE = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]

ROOK_TABLE = [
    0,
    0,
    0,
    5,
    5,
    0,
    0,
    0,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

QUEEN_TABLE = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]

KING_TABLE = [
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _evaluate_position(board):
    """
    Evaluate board position in centipawns from white's perspective.
    Returns a float score (positive = white advantage).
    """
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for piece_type in PIECE_VALUES:
        table = PIECE_TABLES[piece_type]
        for sq in board.pieces(piece_type, chess.WHITE):
            score += PIECE_VALUES[piece_type] + table[sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= PIECE_VALUES[piece_type] + table[chess.square_mirror(sq)]

    return score / 100.0  # Convert to pawns


def _find_best_eval(board):
    """Find the best move's evaluation using a shallow search."""
    best = -9999
    for move in board.legal_moves:
        board.push(move)
        val = -_evaluate_position(board)
        if board.turn == chess.BLACK:
            val = -val
        board.pop()
        best = max(best, val)
    return best


def _classify_move(eval_drop):
    """Classify a move based on how much evaluation was lost."""
    if eval_drop <= -0.1:
        # Player's move was better than expected
        return "brilliant", "Brilliant!", 2
    elif eval_drop <= 0.3:
        return "good", "Good Move", 1
    elif eval_drop <= 0.8:
        return "ok", "OK", 0
    elif eval_drop <= 1.5:
        return "inaccuracy", "Inaccuracy", -1
    elif eval_drop <= 3.0:
        return "mistake", "Mistake", -2
    else:
        return "blunder", "Blunder!", -3


def _describe_move(board, last_move, category):
    """Generate a coaching description for the move."""
    piece = board.piece_at(last_move.to_square)
    piece_name = chess.piece_name(piece.piece_type).capitalize() if piece else "Piece"
    move_san = (
        board.san(last_move) if last_move in board.legal_moves else str(last_move)
    )

    descriptions = {
        "brilliant": [
            f"Excellent {piece_name} move! This improves your position significantly.",
            f"Strong play! You found a powerful move here.",
            f"Great tactical awareness with {piece_name}!",
        ],
        "good": [
            f"Solid {piece_name} move. Keeps your position healthy.",
            f"Good choice. This maintains your advantage.",
            f"Well played. Your position remains strong.",
        ],
        "ok": [
            f"Reasonable move, but there might have been something sharper.",
            f"This is fine, but look for more active possibilities.",
            f"Acceptable, though you could have been more ambitious.",
        ],
        "inaccuracy": [
            f"Slightly imprecise. Consider piece activity and center control.",
            f"This lets your opponent improve. Look for more forcing moves.",
            f"Not the best. Try to look for checks, captures, and threats first.",
        ],
        "mistake": [
            f"This weakens your position. Always check for tactics before moving.",
            f"Careful! This move gives your opponent too much. Look for tactical patterns.",
            f"This costs material or position. Try to calculate one move deeper.",
        ],
        "blunder": [
            f"Major error! Always look for your opponent's threats before moving.",
            f"This loses significant material. Remember: checks, captures, threats!",
            f"Critical mistake. Slow down and consider what your opponent can do.",
        ],
    }

    import random

    msgs = descriptions.get(category, descriptions["ok"])
    return random.choice(msgs)


def evaluate_move(pgn, prev_eval):
    """
    Evaluate the player's latest move and return coaching feedback.

    Args:
        pgn: PGN string of the game so far (includes the player's move)
        prev_eval: evaluation score before the player's move

    Returns:
        dict with coaching feedback:
        - category: brilliant/good/ok/inaccuracy/mistake/blunder
        - badge: display label
        - message: coaching text
        - eval_score: current position evaluation
        - score_delta: points earned/lost
    """
    if not pgn:
        return {
            "category": "system",
            "badge": None,
            "message": "Game started. Good luck!",
            "eval_score": 0.0,
            "score_delta": 0,
        }

    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            return _default_response(prev_eval)

        # Walk to the end of the game
        node = game.end()
        board = node.board()

        # Current evaluation (from white's perspective)
        current_eval = _evaluate_position(board)

        # The player plays white, so eval_drop = how much white's position worsened
        # after the player's move compared to the previous position
        eval_drop = prev_eval - current_eval

        category, badge, score_delta = _classify_move(eval_drop)
        message = _describe_move(board, node.move, category)

        return {
            "category": category,
            "badge": badge,
            "message": message,
            "eval_score": round(current_eval, 1),
            "score_delta": score_delta,
        }

    except Exception as e:
        print(f"Coaching error: {e}")
        return _default_response(prev_eval)


def _default_response(prev_eval):
    return {
        "category": "ok",
        "badge": "OK",
        "message": "Move registered.",
        "eval_score": prev_eval,
        "score_delta": 0,
    }
