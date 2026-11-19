"""Monotone PIT recalibration maps (Gneiting-Balabdaoui-Raftery probabilistic calibration).

If realised PITs u = F(y) have cdf G (not uniform), the recalibrated forecast F' = G o F is
calibrated. For samples we apply it through quantiles: a draw at forecast level u is moved to the
forecast's own quantile at level G^{-1}(u). Applied per cell to the draws' ranks with one common
u per draw, so the copula (dependence across cells) is untouched (Sklar).

A map is stored as knots of G^{-1} on a fixed grid of u, fitted by isotonic smoothing of the
empirical PIT quantiles and shrunk toward the identity with pseudo-count n0.
"""

from __future__ import annotations

import numpy as np

U_GRID = np.array([0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                   0.975, 0.99, 1.0])


def isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators on equal weights."""
    y = list(map(float, y))
    vals, wts = [], []
    for v in y:
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [v2]
            wts[-2:] = [w]
    out = []
    for v, w in zip(vals, wts):
        out += [v] * int(round(w))
    return np.asarray(out)


def fit_inverse_pit(pits: np.ndarray, n0: float = 200.0) -> list[float]:
    pits = np.clip(np.asarray(pits, dtype=float), 0.0, 1.0)
    pits = pits[np.isfinite(pits)]
    if len(pits) < 50:
        return U_GRID.tolist()
    emp = np.quantile(pits, U_GRID)  # G^{-1}(u) estimated
    emp = isotonic_increasing(emp)
    w = len(pits) / (len(pits) + n0)
    knots = w * emp + (1 - w) * U_GRID
    knots[0], knots[-1] = 0.0, 1.0
    return np.maximum.accumulate(knots).tolist()


def apply_inverse_pit(draws: np.ndarray, cells: list[int], knots: list[float]) -> np.ndarray:
    k = np.asarray(knots, dtype=float)
    if np.allclose(k, U_GRID):
        return draws
    out = draws.copy()
    n = draws.shape[0]
    for j in cells:
        col = draws[:, j]
        order = np.argsort(col, kind="stable")
        u = (np.arange(n) + 0.5) / n
        u2 = np.interp(u, U_GRID, k)
        srt = col[order]
        newv = np.quantile(srt, np.clip(u2, 0, 1))
        out[order, j] = newv
    return out
