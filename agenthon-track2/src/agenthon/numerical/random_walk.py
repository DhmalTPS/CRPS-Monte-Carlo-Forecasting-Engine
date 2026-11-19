"""L5 Tier-0 — the exact M0 replica and the minimal fail-safe walk.

`m0_forecast` rebuilds the official text-blind baseline to the letter of docs/M0-BASELINE.md:
trailing 300 observations, steps by target type, gap rule, date-intersection alignment, sample
mean/covariance, anchor at the last observation (0 for log_return), path covariance
min(s_i, s_j) * Sigma, 1e-10 diagonal jitter, Cholesky of cov + 1e-9 I (diagonal fallback),
500 draws from default_rng(crc32(unit_id) & 0x7FFFFFFF), cells sorted by asset then horizon.

Why it matters twice:
1. It is the *fail-safe*. Submitting M0's own draws scores exactly 1.0 by definition of the
   normalisation -- strictly better than any other "safe" forecast (the reference CLI averages
   ~1.29). A card can therefore never cost more than the baseline when anything else breaks.
2. It is the *reference* the score-aware decision layer evaluates candidate forecasts against:
   we can compute S_M0(y) for any hypothetical outcome y.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..timeutil import intervals_days
from ..types import AssetHistory, Card, Grid
from .transform import looks_like_returns


@dataclass
class M0Forecast:
    draws: np.ndarray  # [500, d] in grid (M0) order
    mean: np.ndarray  # [d]
    cov: np.ndarray  # [d, d]
    mu: dict[str, float]
    sigma: np.ndarray  # step covariance in sorted-asset order
    n_rows: int


def m0_seed(unit_id: str) -> int:
    return zlib.crc32(unit_id.encode("utf-8")) & 0x7FFFFFFF


def _m0_steps(s: pd.Series, target_type: str) -> pd.Series:
    win = s.sort_index().iloc[-300:]
    if len(win) < 2:
        return pd.Series(dtype=float)
    iv = intervals_days(win.index)
    lim = max(10.0 * float(np.median(iv)), 5.0)
    if target_type == "log_return":
        if not looks_like_returns(win):
            raise ModelError("M0 tripwire: log_return panel does not look like per-step returns")
        vals = win.to_numpy(dtype=float)[1:]  # row 0 has no preceding interval to test
    else:
        vals = np.diff(win.to_numpy(dtype=float))
    out = pd.Series(vals, index=win.index[1:])
    return out[iv <= lim]


def m0_forecast(card: Card, grid: Grid, histories: dict[str, AssetHistory],
                n_draws: int = 500) -> M0Forecast:
    assets = sorted({c.asset for c in grid.cells})
    steps = {}
    for a in assets:
        h = histories.get(a)
        if h is None:
            raise ModelError(f"M0: no history for {a}")
        steps[a] = _m0_steps(h.series, card.target_type)
    frame = pd.DataFrame(steps).dropna()  # intersection of dates present for every asset
    if len(frame) < 2:
        raise ModelError("M0: fewer than 2 aligned steps")
    mu = frame.mean().to_numpy(dtype=float)
    sigma = np.atleast_2d(np.cov(frame.to_numpy(dtype=float), rowvar=False))
    idx = {a: i for i, a in enumerate(assets)}
    anchor = {a: (0.0 if card.target_type == "log_return" else float(histories[a].series.iloc[-1]))
              for a in assets}
    d = grid.d
    mean = np.empty(d)
    cov = np.empty((d, d))
    for i, ci in enumerate(grid.cells):
        ai = idx[ci.asset]
        mean[i] = anchor[ci.asset] + ci.steps * mu[ai]
        for j, cj in enumerate(grid.cells):
            cov[i, j] = min(ci.steps, cj.steps) * sigma[ai, idx[cj.asset]]
    cov[np.diag_indices(d)] += 1e-10
    try:
        chol = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(cov + 1e-9 * np.eye(d)), 0.0, None)))
    rng = np.random.default_rng(m0_seed(card.unit_id))
    z = rng.standard_normal((n_draws, d))
    draws = mean + z @ chol.T
    if not np.all(np.isfinite(draws)):
        raise ModelError("M0: non-finite draws")
    return M0Forecast(draws=draws, mean=mean, cov=cov, mu={a: float(mu[idx[a]]) for a in assets},
                      sigma=sigma, n_rows=int(len(frame)))


def minimal_walk(card: Card, grid: Grid, histories: dict[str, AssetHistory], n_draws: int,
                 seed: int) -> np.ndarray:
    """Last-resort forecast when even M0 cannot be rebuilt. Cannot raise on sane inputs.

    Independent-increment Gaussian walk from the anchor with a robust (MAD) step scale and a common
    shock across assets (perfect correlation is a safer joint default than independence for the
    variogram when nothing is known). A positive floor on the spread keeps g3 satisfied.
    """
    rng = np.random.default_rng(seed)
    z_common = rng.standard_normal(n_draws)
    out = np.empty((n_draws, grid.d))
    for i, c in enumerate(grid.cells):
        s = histories[c.asset].series if c.asset in histories else pd.Series([0.0, 0.0])
        v = s.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if card.target_type == "log_return":
            st = v
            anchor = 0.0
        else:
            st = np.diff(v) if len(v) > 1 else np.array([0.0])
            anchor = float(v[-1]) if len(v) else 0.0
        st = st[-750:]
        mad = float(np.median(np.abs(st - np.median(st)))) * 1.4826 if len(st) else 0.0
        sd = max(mad, float(np.std(st)) if len(st) else 0.0, 1e-6 * max(abs(anchor), 1.0))
        out[:, i] = anchor + z_common * sd * np.sqrt(max(c.steps, 1))
    return out
