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


def top_k_policy(
    policy: ArrayLike,
    legal_mask: ArrayLike | None = None,
    *,
    k: int,
    floor_probability: float = 0.0,
) -> FloatArray:
    """Keep the strongest k legal moves and renormalize the policy plane.

    A tiny floor can be assigned to non-top-k legal moves to avoid a completely
    collapsed target while still concentrating supervision on the teacher/search
    candidates.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if floor_probability < 0:
        raise ValueError("floor_probability must be non-negative")

    values = normalize_policy(policy, legal_mask)
    if legal_mask is None:
        mask = np.ones_like(values, dtype=bool)
    else:
        mask = _as_legal_mask(legal_mask)
        if mask.shape != values.shape:
            raise ValueError("legal_mask shape must match policy shape")

    legal_indices = np.flatnonzero(mask.reshape(-1))
    if legal_indices.size <= k:
        return values

    flat = values.reshape(-1)
    ranked_legal = legal_indices[np.argsort(flat[legal_indices])[::-1]]
    keep = ranked_legal[:k]

    pruned = np.zeros_like(flat, dtype=np.float32)
    if floor_probability > 0:
        legal_floor = np.setdiff1d(legal_indices, keep, assume_unique=False)
        pruned[legal_floor] = floor_probability
    pruned[keep] = flat[keep]

    return normalize_policy(pruned.reshape(values.shape), mask)


def mix_policy_targets(
    primary_policy: ArrayLike,
    secondary_policy: ArrayLike,
    *,
    secondary_weight: float,
    legal_mask: ArrayLike | None = None,
) -> FloatArray:
    """Blend two policy targets and normalize over legal moves."""

    if not 0.0 <= secondary_weight <= 1.0:
        raise ValueError("secondary_weight must be in [0, 1]")

    primary = normalize_policy(primary_policy, legal_mask)
    secondary = normalize_policy(secondary_policy, legal_mask)
    mixed = (1.0 - secondary_weight) * primary + secondary_weight * secondary
    return normalize_policy(mixed, legal_mask)


def mix_value_targets(
    primary_value: ArrayLike,
    secondary_value: ArrayLike,
    *,
    secondary_weight: float,
) -> FloatArray:
    """Blend scalar or vector value targets with a teacher/search value."""

    if not 0.0 <= secondary_weight <= 1.0:
        raise ValueError("secondary_weight must be in [0, 1]")

    primary = np.asarray(primary_value, dtype=np.float32)
    secondary = np.asarray(secondary_value, dtype=np.float32)
    if primary.shape != secondary.shape:
        raise ValueError("primary_value and secondary_value shapes must match")
    return ((1.0 - secondary_weight) * primary + secondary_weight * secondary).astype(
        np.float32,
        copy=False,
    )
