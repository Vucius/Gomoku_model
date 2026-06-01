"""Feature-plane encoding for Gomoku positions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float32]

CHANNELS: tuple[str, ...] = (
    "current_player_stones",
    "opponent_stones",
    "legal_moves",
    "edge_distance",
    "x_coord",
    "y_coord",
    "center_distance",
    "phase",
)


def _board_array(board: ArrayLike) -> NDArray[np.integer | np.floating]:
    board_array = np.asarray(board)
    if board_array.ndim != 2:
        raise ValueError(f"board must be a 2D array, got shape {board_array.shape}")
    if 0 in board_array.shape:
        raise ValueError("board dimensions must be non-zero")
    return board_array


def edge_distance_plane(
    height: int,
    width: int,
    *,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Return normalized distance from the closest board edge.

    Corners and edge cells are 0. The most central line/cell is 1 when a board
    has an interior. A 1-wide board returns all zeros.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    y = np.arange(height, dtype=dtype)[:, None]
    x = np.arange(width, dtype=dtype)[None, :]
    distance = np.minimum(np.minimum(y, x), np.minimum(height - 1 - y, width - 1 - x))
    max_distance = float(distance.max())
    if max_distance > 0:
        distance = distance / max_distance
    return distance.astype(dtype, copy=False)


def coordconv_planes(
    height: int,
    width: int,
    *,
    dtype: np.dtype[np.float32] = np.float32,
) -> tuple[FloatArray, FloatArray]:
    """Return x/y coordinate planes normalized to [-1, 1]."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    if width == 1:
        x_axis = np.zeros((1,), dtype=dtype)
    else:
        x_axis = np.linspace(-1.0, 1.0, width, dtype=dtype)

    if height == 1:
        y_axis = np.zeros((1,), dtype=dtype)
    else:
        y_axis = np.linspace(-1.0, 1.0, height, dtype=dtype)

    x_plane = np.broadcast_to(x_axis[None, :], (height, width)).astype(dtype, copy=True)
    y_plane = np.broadcast_to(y_axis[:, None], (height, width)).astype(dtype, copy=True)
    return x_plane, y_plane


def center_distance_plane(
    height: int,
    width: int,
    *,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Return a center-proximity plane normalized to [0, 1]."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    y = np.arange(height, dtype=dtype)[:, None]
    x = np.arange(width, dtype=dtype)[None, :]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2, dtype=dtype)
    max_distance = float(distance.max())
    if max_distance == 0:
        return np.ones((height, width), dtype=dtype)
    return (1.0 - distance / max_distance).astype(dtype, copy=False)


def legal_moves_plane(board: ArrayLike, *, dtype: np.dtype[np.float32] = np.float32) -> FloatArray:
    """Return a binary plane where empty cells are legal moves."""

    board_array = _board_array(board)
    return (board_array == 0).astype(dtype)


def phase_plane(
    move_count: int,
    height: int,
    width: int,
    *,
    max_moves: int | None = None,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Return a constant plane containing normalized game phase."""

    if move_count < 0:
        raise ValueError("move_count must be non-negative")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    max_move_count = max_moves if max_moves is not None else height * width
    if max_move_count <= 0:
        raise ValueError("max_moves must be positive")

    normalized = min(float(move_count) / float(max_move_count), 1.0)
    return np.full((height, width), normalized, dtype=dtype)


def encode_position(
    board: ArrayLike,
    current_player: int,
    *,
    move_count: int | None = None,
    include_channels: Sequence[str] = CHANNELS,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Encode a board as channel-first neural-network feature planes.

    The board may use either relative values or absolute colors. The current
    player plane is `board == current_player`; the opponent plane is any
    occupied cell that is not the current player.
    """

    if current_player == 0:
        raise ValueError("current_player must be non-zero")

    board_array = _board_array(board)
    height, width = board_array.shape

    x_plane, y_plane = coordconv_planes(height, width, dtype=dtype)
    channel_values: dict[str, FloatArray] = {
        "current_player_stones": (board_array == current_player).astype(dtype),
        "opponent_stones": ((board_array != 0) & (board_array != current_player)).astype(dtype),
        "legal_moves": legal_moves_plane(board_array, dtype=dtype),
        "edge_distance": edge_distance_plane(height, width, dtype=dtype),
        "x_coord": x_plane,
        "y_coord": y_plane,
        "center_distance": center_distance_plane(height, width, dtype=dtype),
    }
    if move_count is not None:
        channel_values["phase"] = phase_plane(move_count, height, width, dtype=dtype)

    missing = [channel for channel in include_channels if channel not in channel_values]
    if missing:
        raise ValueError(
            "requested channels are unavailable: "
            + ", ".join(missing)
            + ". Pass move_count to include phase."
        )

    return np.stack([channel_values[channel] for channel in include_channels]).astype(dtype, copy=False)
