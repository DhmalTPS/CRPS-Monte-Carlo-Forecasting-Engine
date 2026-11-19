"""Exact replicas of the Track-2 metric, vectorised over MANY hypothetical outcomes.

Matches qfbench2_common.scoring.crps (fair ensemble CRPS, variogram p=0.5 with unit weights,
summed over ordered pairs) and qfbench2_track_forecasting.tail.tail_pinball (mean over levels and
cells, numpy's default linear quantile). tests/unit/test_output.py checks parity against the
official package when it is importable.

Shapes: samples [m, d]; outcomes Y [K, d]. Every function returns a length-K vector.
"""

from __future__ import annotations

import numpy as np


def crps_cells(samples: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fair CRPS per (outcome, cell): [K, d]."""
    x = np.sort(np.asarray(samples, dtype=float), axis=0)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m, d = x.shape
    i = np.arange(1, m + 1, dtype=float)[:, None]
    spread = 2.0 * np.sum((2 * i - m - 1) * x, axis=0) / (m * (m - 1) if m > 1 else m * m)
    cs = np.vstack([np.zeros((1, d)), np.cumsum(x, axis=0)])  # [m+1, d]
    out = np.empty((Y.shape[0], d))
    for j in range(d):
        k = np.searchsorted(x[:, j], Y[:, j], side="left")
        below = Y[:, j] * k - cs[k, j]
        above = (cs[m, j] - cs[k, j]) - Y[:, j] * (m - k)
        out[:, j] = (below + above) / m - 0.5 * spread[j]
    return out


def crps_marginal(samples: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return crps_cells(samples, Y).mean(axis=1)


def variogram_stat(samples: np.ndarray, p: float = 0.5) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    return (np.abs(x[:, :, None] - x[:, None, :]) ** p).mean(axis=0)


def variogram(samples: np.ndarray, Y: np.ndarray, p: float = 0.5,
              x_vg: np.ndarray | None = None) -> np.ndarray:
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[1] < 2:
        return np.zeros(Y.shape[0])
    xv = variogram_stat(samples, p) if x_vg is None else x_vg
    yv = np.abs(Y[:, :, None] - Y[:, None, :]) ** p
    return ((yv - xv[None]) ** 2).sum(axis=(1, 2))


def pinball(samples: np.ndarray, Y: np.ndarray,
            levels: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)) -> np.ndarray:
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    q = np.quantile(np.asarray(samples, dtype=float), levels, axis=0)  # [L, d]
    tot = np.zeros(Y.shape[0])
    for li, a in enumerate(levels):
        diff = Y - q[li][None, :]
        loss = np.where(diff >= 0, a * diff, (a - 1.0) * diff)
        tot += loss.mean(axis=1)
    return tot / len(levels)


def components(samples: np.ndarray, Y: np.ndarray, levels=(0.01, 0.05, 0.95, 0.99),
               joint: bool = True) -> np.ndarray:
    """[K, 3] raw (marginal, joint, tail) components."""
    Y = np.atleast_2d(Y)
    vg = variogram(samples, Y) if (joint and Y.shape[1] > 1) else np.zeros(Y.shape[0])
    return np.column_stack([crps_marginal(samples, Y), vg, pinball(samples, Y, levels)])


def normalised(comp: np.ndarray, comp_m0: np.ndarray, weights: tuple[float, float, float],
               clip: float = 4.0) -> np.ndarray:
    """Per-outcome normalised composite exactly as the leaderboard computes it.

    A reference component <= 0 is stored as 1.0 (not normalised), per docs/M0-BASELINE.md sec. 5.
    """
    ref = np.where(comp_m0 > 0, comp_m0, 1.0)
    w = np.asarray(weights, dtype=float)
    s = (comp / ref * w[None, :]).sum(axis=1)
    return np.minimum(s, clip)
