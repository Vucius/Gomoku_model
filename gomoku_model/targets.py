"""Policy target shaping utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .features import FloatArray
from .sampling import edge_distance_for_coord


def _as_float_plane(values: ArrayLike) -> FloatArray:
    plane = np.asarray(values, dtype=np.float32)
    if plane.ndim != 2:
        raise ValueError(f"policy values must be a 2D plane, got shape {plane.shape}")
    return plane


def _as_legal_mask(legal_mask: ArrayLike) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"legal_mask must be a 2D plane, got shape {mask.shape}")
    if not np.any(mask):
        raise ValueError("legal_mask must contain at least one legal move")
    return mask


def policy_from_move(
    move_coord: tuple[int, int] | np.ndarray,
    height: int,
    width: int,
    *,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Create a one-hot policy plane from an (x, y) move coordinate."""

    edge_distance_for_coord(move_coord, height, width)
    x, y = int(move_coord[0]), int(move_coord[1])
    policy = np.zeros((height, width), dtype=dtype)
    policy[y, x] = 1.0
    return policy


def legal_uniform_policy(
    legal_mask: ArrayLike,
    *,
    dtype: np.dtype[np.float32] = np.float32,
) -> FloatArray:
    """Return a uniform distribution over legal moves."""

    mask = _as_legal_mask(legal_mask)
    policy = mask.astype(dtype)
    return policy / policy.sum(dtype=dtype)


def normalize_policy(policy: ArrayLike, legal_mask: ArrayLike | None = None) -> FloatArray:
    """Normalize a non-negative policy plane, optionally masking illegal cells."""

    values = _as_float_plane(policy).copy()
    if np.any(values < 0):
        raise ValueError("policy values must be non-negative")

    if legal_mask is not None:
        mask = _as_legal_mask(legal_mask)
        if mask.shape != values.shape:
            raise ValueError("legal_mask shape must match policy shape")
        values[~mask] = 0.0

    total = float(values.sum())
    if total <= 0:
        if legal_mask is None:
            raise ValueError("policy must have positive total probability")
        return legal_uniform_policy(legal_mask)
    return (values / total).astype(np.float32, copy=False)


def label_smooth_policy(policy: ArrayLike, legal_mask: ArrayLike, epsilon: float) -> FloatArray:
    """Blend a policy target with a uniform distribution over legal moves."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    normalized_policy = normalize_policy(policy, legal_mask)
    uniform = legal_uniform_policy(legal_mask)
    return ((1.0 - epsilon) * normalized_policy + epsilon * uniform).astype(np.float32, copy=False)


def soften_visit_counts(
    visit_counts: ArrayLike,
    *,
    temperature: float,
    legal_mask: ArrayLike | None = None,
) -> FloatArray:
    """Convert visit counts to a temperature-softened policy distribution."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    counts = _as_float_plane(visit_counts).copy()
    if np.any(counts < 0):
        raise ValueError("visit_counts must be non-negative")

    if legal_mask is not None:
        mask = _as_legal_mask(legal_mask)
        if mask.shape != counts.shape:
            raise ValueError("legal_mask shape must match visit_counts shape")
        counts[~mask] = 0.0

    if float(counts.sum()) <= 0:
        if legal_mask is None:
            raise ValueError("visit_counts must have positive total when no legal_mask is supplied")
        return legal_uniform_policy(legal_mask)

    softened = np.power(counts, 1.0 / temperature, dtype=np.float32)
    return normalize_policy(softened, legal_mask)
