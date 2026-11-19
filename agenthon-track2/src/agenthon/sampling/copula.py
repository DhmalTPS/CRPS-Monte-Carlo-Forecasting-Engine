"""L21 — copula/mixing variables.

A multivariate Student-t is a Gaussian copula-free way to get (i) fat marginal tails and
(ii) tail dependence with one shared scale per draw: X = Z * sqrt(W), W = (nu - 2) / chi2_nu,
so Var(X) = Var(Z) exactly (the variance forecast is not distorted by the tail choice).
By Sklar's theorem the marginals can then be remapped monotonically (calibration) without
touching this dependence structure.
"""

from __future__ import annotations

import numpy as np


def t_mixing(rng: np.random.Generator, n: int, df: float) -> np.ndarray:
    """Variance-preserving scale-mixing weights sqrt(W) for a multivariate-t with `df` dof."""
    if not np.isfinite(df) or df > 200:
        return np.ones(n)
    df = max(float(df), 2.5)
    chi = rng.chisquare(df, size=n)
    return np.sqrt((df - 2.0) / np.maximum(chi, 1e-12))


def soft_floor(y: np.ndarray, floor: float, k: float = 0.25) -> np.ndarray:
    """Smoothly compress values below floor + k toward `floor` (C1 continuous, slope 1 at the knee).

    Used for USD yields near the effective lower bound: a Gaussian walk from 0.1% puts real mass
    on -0.5% yields that never print; this keeps the upper tail untouched.
    """
    b = floor + k
    out = np.asarray(y, dtype=float).copy()
    m = out < b
    out[m] = b - k * np.log1p((b - out[m]) / k)
    return np.maximum(out, floor - 5 * k)  # log compression is unbounded below; hard stop
