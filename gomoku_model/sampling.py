"""Dataset loading and edge-aware sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .features import FloatArray, encode_position

IntArray = NDArray[np.integer]

EDGE_BUCKETS: tuple[str, ...] = ("edge_0", "edge_1", "edge_2", "center")
DEFAULT_BUCKET_WEIGHTS: dict[str, float] = {
    "edge_0": 0.20,
    "edge_1": 0.20,
    "edge_2": 0.20,
    "center": 0.40,
}


def edge_distance_for_coord(coord: NDArray[np.integer] | tuple[int, int], height: int, width: int) -> int:
    """Return min distance from an (x, y) coordinate to the board edge."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    x, y = int(coord[0]), int(coord[1])
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"move coordinate {(x, y)} is outside a {width}x{height} board")

    return min(x, y, width - 1 - x, height - 1 - y)


def edge_bucket_for_coord(coord: NDArray[np.integer] | tuple[int, int], height: int, width: int) -> str:
    """Map an (x, y) move coordinate into an edge-distance bucket."""

    distance = edge_distance_for_coord(coord, height, width)
    if distance <= 2:
        return f"edge_{distance}"
    return "center"


def bucket_indices(next_moves_coords: NDArray[np.integer], height: int, width: int) -> dict[str, IntArray]:
    """Group sample indices by the next move's edge-distance bucket."""

    coords = np.asarray(next_moves_coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"next_moves_coords must have shape (N, 2), got {coords.shape}")

    buckets: dict[str, list[int]] = {bucket: [] for bucket in EDGE_BUCKETS}
    for index, coord in enumerate(coords):
        buckets[edge_bucket_for_coord(coord, height, width)].append(index)

    return {bucket: np.asarray(indices, dtype=np.int64) for bucket, indices in buckets.items()}


def sample_balanced_indices(
    next_moves_coords: NDArray[np.integer],
    height: int,
    width: int,
    batch_size: int,
    *,
    bucket_weights: Mapping[str, float] = DEFAULT_BUCKET_WEIGHTS,
    rng: np.random.Generator | None = None,
) -> IntArray:
    """Sample indices using edge-aware bucket weights.

    Buckets with no samples are skipped and their weight is redistributed to
    available buckets. Sampling uses replacement only when a bucket has fewer
    examples than its requested count.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    grouped = bucket_indices(next_moves_coords, height, width)
    available = {bucket: indices for bucket, indices in grouped.items() if len(indices) > 0}
    if not available:
        raise ValueError("cannot sample from an empty move list")

    for bucket, weight in bucket_weights.items():
        if bucket not in EDGE_BUCKETS:
            raise ValueError(f"unknown bucket: {bucket}")
        if weight < 0:
            raise ValueError("bucket weights must be non-negative")

    active_weight_sum = sum(bucket_weights.get(bucket, 0.0) for bucket in available)
    if active_weight_sum <= 0:
        raise ValueError("at least one available bucket must have positive weight")

    generator = rng if rng is not None else np.random.default_rng()
    exact_counts = {
        bucket: batch_size * bucket_weights.get(bucket, 0.0) / active_weight_sum for bucket in available
    }
    counts = {bucket: int(np.floor(count)) for bucket, count in exact_counts.items()}
    remaining = batch_size - sum(counts.values())
    remainder_order = sorted(
        exact_counts,
        key=lambda bucket: (exact_counts[bucket] - counts[bucket], bucket),
        reverse=True,
    )
    for bucket in remainder_order[:remaining]:
        counts[bucket] += 1

    selected: list[IntArray] = []
    for bucket, count in counts.items():
        if count == 0:
            continue
        indices = available[bucket]
        selected.append(generator.choice(indices, size=count, replace=len(indices) < count))

    result = np.concatenate(selected).astype(np.int64, copy=False)
    generator.shuffle(result)
    return result


@dataclass(frozen=True)
class FullBoardDataset:
    """Numpy-backed full-board dataset with feature encoding helpers."""

    board_states: NDArray[np.integer]
    next_moves_coords: NDArray[np.integer]
    next_moves_players: NDArray[np.integer]

    @classmethod
    def from_directory(cls, directory: str | Path, *, mmap_mode: str | None = "r") -> "FullBoardDataset":
        data_dir = Path(directory)
        return cls(
            board_states=np.load(data_dir / "board_states.npy", mmap_mode=mmap_mode),
            next_moves_coords=np.load(data_dir / "next_moves_coords.npy", mmap_mode=mmap_mode),
            next_moves_players=np.load(data_dir / "next_moves_players.npy", mmap_mode=mmap_mode),
        )

    @classmethod
    def from_split(
        cls,
        dataset_root: str | Path,
        split: str = "train",
        *,
        mmap_mode: str | None = "r",
    ) -> "FullBoardDataset":
        root = Path(dataset_root)
        if split in {"train", "test"}:
            return cls.from_directory(root / split / "full_board", mmap_mode=mmap_mode)
        if split == "full":
            return cls.from_directory(root / "full", mmap_mode=mmap_mode)
        raise ValueError("split must be one of: train, test, full")

    def __post_init__(self) -> None:
        if self.board_states.ndim != 3:
            raise ValueError(f"board_states must have shape (N, H, W), got {self.board_states.shape}")
        if self.next_moves_coords.shape != (len(self.board_states), 2):
            raise ValueError("next_moves_coords must have shape (N, 2) matching board_states")
        if self.next_moves_players.shape != (len(self.board_states),):
            raise ValueError("next_moves_players must have shape (N,) matching board_states")

    def __len__(self) -> int:
        return int(self.board_states.shape[0])

    @property
    def board_shape(self) -> tuple[int, int]:
        return int(self.board_states.shape[1]), int(self.board_states.shape[2])

    def encode(self, index: int) -> FloatArray:
        board = self.board_states[index]
        player = int(self.next_moves_players[index])
        return encode_position(board, player, move_count=int(np.count_nonzero(board)))

    def sample_balanced_batch(
        self,
        batch_size: int,
        *,
        bucket_weights: Mapping[str, float] = DEFAULT_BUCKET_WEIGHTS,
        rng: np.random.Generator | None = None,
    ) -> tuple[IntArray, FloatArray, IntArray, IntArray]:
        height, width = self.board_shape
        indices = sample_balanced_indices(
            self.next_moves_coords,
            height,
            width,
            batch_size,
            bucket_weights=bucket_weights,
            rng=rng,
        )
        features = np.stack([self.encode(int(index)) for index in indices])
        return (
            indices,
            features,
            np.asarray(self.next_moves_coords[indices]),
            np.asarray(self.next_moves_players[indices]),
        )
