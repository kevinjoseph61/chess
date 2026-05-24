"""
Board encoding utilities for the AlphaZero chess network.

Converts python-chess Board objects to/from tensor representations,
and handles move encoding/decoding (AlphaZero 8x8x73 scheme).
"""

import chess
import numpy as np
import torch

# ---- Board → Tensor ----

PIECE_PLANES = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as an 18x8x8 numpy array (float32).

    Planes 0-11:  piece positions (from white's perspective if white to move,
                  flipped if black to move so the model always sees "self" pieces first)
    Planes 12-15: castling rights (K, Q, k, q)
    Plane 16:     en passant square
    Plane 17:     side to move (all 1s = current player is white from canonical view)
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # If black to move, we flip the board so the network always sees from
    # the perspective of the side to move
    flip = not board.turn  # flip if black

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        row = chess.square_rank(sq)
        col = chess.square_file(sq)

        if flip:
            row = 7 - row
            # Swap colors in the plane index
            color = not piece.color
        else:
            color = piece.color

        plane = PIECE_PLANES[(piece.piece_type, color)]
        tensor[plane, row, col] = 1.0

    # Castling rights
    castling_map = [
        (chess.BB_H1, 12),  # White kingside
        (chess.BB_A1, 13),  # White queenside
        (chess.BB_H8, 14),  # Black kingside
        (chess.BB_A8, 15),  # Black queenside
    ]
    for bb, plane_idx in castling_map:
        if board.castling_rights & bb:
            tensor[plane_idx, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        if flip:
            ep_row = 7 - ep_row
        tensor[16, ep_row, ep_col] = 1.0

    # Side to move (1 if the canonical view player is white)
    tensor[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor


# ---- Move encoding: AlphaZero 8x8x73 scheme ----

# Direction offsets for queen-type moves: (row_delta, col_delta)
# Order: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# Knight move offsets
KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

# Underpromotion directions: (row_delta, col_delta) for pawn on 7th rank
PROMO_DIRS = [(-1, -1), (-1, 0), (-1, 1)]  # left capture, forward, right capture
PROMO_PIECES = [
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
]  # queen promotion = default queen move


def _move_to_plane(from_sq, to_sq, promotion, flip):
    """Return the plane index (0-72) for a move."""
    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    to_row = chess.square_rank(to_sq)
    to_col = chess.square_file(to_sq)

    if flip:
        from_row = 7 - from_row
        to_row = 7 - to_row

    dr = to_row - from_row
    dc = to_col - from_col

    # Check underpromotion
    if promotion is not None and promotion != chess.QUEEN:
        for dir_idx, (pdr, pdc) in enumerate(PROMO_DIRS):
            if dr == pdr and dc == pdc:
                piece_idx = PROMO_PIECES.index(promotion)
                return 64 + dir_idx * 3 + piece_idx  # planes 64-72
        # Shouldn't reach here, but fall through to queen move encoding

    # Check knight move
    for i, (kr, kc) in enumerate(KNIGHT_MOVES):
        if dr == kr and dc == kc:
            return 56 + i  # planes 56-63

    # Queen-type move (including queen promotion which is encoded as a regular queen move)
    if dr == 0 and dc == 0:
        return 0  # null move, shouldn't happen

    # Determine direction
    if dr != 0:
        norm_dr = dr // abs(dr)
    else:
        norm_dr = 0
    if dc != 0:
        norm_dc = dc // abs(dc)
    else:
        norm_dc = 0

    distance = max(abs(dr), abs(dc))

    for dir_idx, (qdr, qdc) in enumerate(QUEEN_DIRS):
        if norm_dr == qdr and norm_dc == qdc:
            return dir_idx * 7 + (distance - 1)  # planes 0-55

    raise ValueError(f"Cannot encode move {from_sq}->{to_sq} dr={dr} dc={dc}")


def move_to_index(move: chess.Move, flip: bool) -> int:
    """
    Convert a chess.Move to a flat index in [0, 4671].
    If flip=True, the board perspective is flipped (for black moves).
    """
    from_sq = move.from_square
    to_sq = move.to_square

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)

    if flip:
        from_row = 7 - from_row

    plane = _move_to_plane(from_sq, to_sq, move.promotion, flip)
    return from_row * 8 * 73 + from_col * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert a flat index back to a chess.Move.
    Validates against legal moves.
    """
    flip = not board.turn

    # Find the closest legal move matching this index
    for move in board.legal_moves:
        if move_to_index(move, flip) == index:
            return move

    return chess.Move.null()


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a binary mask of shape (4672,) for legal moves."""
    flip = not board.turn
    mask = np.zeros(4672, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move, flip)
        mask[idx] = 1.0
    return mask


def get_move_indices(board: chess.Board) -> dict:
    """Return a dict mapping move index -> chess.Move for all legal moves."""
    flip = not board.turn
    result = {}
    for move in board.legal_moves:
        idx = move_to_index(move, flip)
        result[idx] = move
    return result
