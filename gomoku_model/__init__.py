"""Utilities for Gomoku neural-network training."""

from .features import (
    CHANNELS,
    center_distance_plane,
    coordconv_planes,
    edge_distance_plane,
    encode_position,
    legal_moves_plane,
    phase_plane,
)
from .sampling import (
    DEFAULT_BUCKET_WEIGHTS,
    EDGE_BUCKETS,
    FullBoardDataset,
    bucket_indices,
    edge_bucket_for_coord,
    edge_distance_for_coord,
    sample_balanced_indices,
)
from .targets import (
    label_smooth_policy,
    legal_uniform_policy,
    normalize_policy,
    policy_from_move,
    soften_visit_counts,
)

__all__ = [
    "CHANNELS",
    "DEFAULT_BUCKET_WEIGHTS",
    "EDGE_BUCKETS",
    "FullBoardDataset",
    "bucket_indices",
    "center_distance_plane",
    "coordconv_planes",
    "edge_bucket_for_coord",
    "edge_distance_plane",
    "edge_distance_for_coord",
    "encode_position",
    "label_smooth_policy",
    "legal_moves_plane",
    "legal_uniform_policy",
    "normalize_policy",
    "phase_plane",
    "policy_from_move",
    "sample_balanced_indices",
    "soften_visit_counts",
]
