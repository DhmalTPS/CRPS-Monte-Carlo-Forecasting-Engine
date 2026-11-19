"""Draw-matrix sanitation between stages. Raises instead of 'repairing' anything substantive."""

from __future__ import annotations

import numpy as np

from ..errors import ModelError


def check_draws(draws: np.ndarray, d: int, what: str = "draws") -> np.ndarray:
    a = np.asarray(draws, dtype=float)
    if a.ndim != 2 or a.shape[1] != d or a.shape[0] < 2:
        raise ModelError(f"{what}: bad shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ModelError(f"{what}: non-finite values")
    sd = a.std(axis=0)
    if np.any(sd <= 0):
        raise ModelError(f"{what}: zero-spread cell")
    return a


def spread_sanity(draws: np.ndarray, ref: np.ndarray, lo: float = 0.15, hi: float = 8.0) -> bool:
    """Every cell's sd within [lo, hi] x the reference (M0) sd: catches unit/scale blunders."""
    s = draws.std(axis=0)
    r = np.maximum(ref.std(axis=0), 1e-12)
    ratio = s / r
    return bool(np.all(ratio >= lo) and np.all(ratio <= hi))
