"""Historical analogue engine (Conditional, P2) — same unit, same series, past only (Correction C5).

For the current vol-standardised state (vol ratio, 20d and 63d trend z-scores) find the k most
similar past dates of the SAME series (Mahalanobis distance on those features, covariance
estimated on the unit's own history), look at their subsequent s-step moves standardised by the
volatility known at that past date, and report the analogue scale multiplier
sd(standardised outcomes) per step count. It is a *vote on the width*, never a standalone forecast,
and it is disabled unless the walk-forward ablation shows a robust gain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..types import AssetHistory
from .statistical import ewma_var_path
from .transform import level_steps, return_steps


def _features(x: np.ndarray, halflife: float) -> tuple[np.ndarray, np.ndarray]:
    ew = np.sqrt(np.maximum(ewma_var_path(x, halflife), 1e-24))
    lr = np.sqrt(np.maximum(pd.Series(x * x).rolling(750, min_periods=250).mean().to_numpy(), 1e-24))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    idx = np.arange(n)
    m20 = (cs[idx + 1] - cs[np.maximum(idx + 1 - 20, 0)]) / (ew * np.sqrt(20))
    m63 = (cs[idx + 1] - cs[np.maximum(idx + 1 - 63, 0)]) / (ew * np.sqrt(63))
    f = np.column_stack([np.log(ew / lr), m20, m63])
    return f, ew


def analogue_scale(h: AssetHistory, target_type: str, step_points: list[int],
                   cfg: dict[str, Any]) -> np.ndarray | None:
    acfg = cfg["analogue"]
    st = return_steps(h.series) if target_type == "log_return" else level_steps(h.series)
    x = st.to_numpy(dtype=float)
    if len(x) < int(acfg.get("min_history", 750)) or h.is_transfer:
        return None
    f, ew = _features(x, float(cfg["numerical"]["ewma_halflife"]))
    smax = max(step_points)
    cand = np.arange(260, len(x) - smax - 1)
    if len(cand) < 200:
        return None
    F = f[cand]
    ok = np.all(np.isfinite(F), axis=1)
    cand, F = cand[ok], F[ok]
    cur = f[-1]
    if not np.all(np.isfinite(cur)) or len(cand) < 200:
        return None
    cov = np.cov(F, rowvar=False) + 1e-6 * np.eye(F.shape[1])
    inv = np.linalg.inv(cov)
    dlt = F - cur
    dist = np.einsum("ij,jk,ik->i", dlt, inv, dlt)
    k = min(int(acfg.get("k", 40)), len(cand) // 5)
    near = cand[np.argsort(dist)[:k]]
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = []
    for s in step_points:
        mv = (cs[near + 1 + s] - cs[near + 1]) / (ew[near] * np.sqrt(s))
        allmv = (cs[cand + 1 + s] - cs[cand + 1]) / (ew[cand] * np.sqrt(s))
        out.append(float(np.std(mv) / max(np.std(allmv), 1e-12)))
    return np.asarray(out)
