"""L19 — joint dependence: cross-asset correlation updates that provably stay valid.

Proposition 3 of the design document: if Sigma_0, Sigma_text are PSD then
(1 - g) Sigma_0 + g Sigma_text is PSD for g in [0, 1] (convex cone). A text "risk-off" signal is
expressed as a convex move toward a one-factor structure with higher common loading, never as an
element-wise edit that could leave the cone. `nearest_correlation` is a final guard only.
"""

from __future__ import annotations

import numpy as np

from ..numerical.state import nearest_correlation


def stress_correlation(corr: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """One-factor 'stress' version of corr: each |rho_ij| pushed toward 1 with its sign kept.

    Built as a PSD matrix: rho_ij^stress = s_i s_j * r where s_i is the sign of asset i's loading
    on the first principal component and r = mean |rho| raised toward 1 by `intensity`.
    """
    n = corr.shape[0]
    if n == 1:
        return corr.copy()
    w, v = np.linalg.eigh(corr)
    pc = v[:, -1]
    sgn = np.where(pc >= 0, 1.0, -1.0)
    off = np.abs(corr[np.triu_indices(n, 1)])
    r = float(np.clip(np.mean(off) + intensity * (1 - np.mean(off)), 0.0, 0.95))
    s = np.outer(sgn, sgn) * r
    np.fill_diagonal(s, 1.0)
    return nearest_correlation(s)


def blend_correlation(corr0: np.ndarray, corr_text: np.ndarray, g: float) -> np.ndarray:
    g = float(np.clip(g, 0.0, 1.0))
    if g == 0.0:
        return corr0.copy()
    return nearest_correlation((1 - g) * corr0 + g * corr_text)


def safe_cholesky(c: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(c + 1e-12 * np.eye(c.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_correlation(c, 1e-6) + 1e-9 * np.eye(c.shape[0]))
