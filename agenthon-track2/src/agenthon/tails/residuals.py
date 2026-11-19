"""L20 — residual bank from an IN-UNIT walk-forward backtest (point-in-time by construction).

For past cutoffs t in the unit's own history (t + s <= last observation <= as-of) the same variance
formulas as the live forecast give V(s | t); the realised s-step move gives a standardised error
z = (move - drift*s) / sqrt(V). The bank {z} measures, for THIS asset, whether our width is
biased (E z^2 != 1) and how fat the horizon-level tails are (kurtosis). Uses only data <= as-of.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..numerical.statistical import aggregate_variance, ewma_var_path, rolling_var
from ..types import AssetHistory, AssetState
from ..numerical.transform import level_steps, return_steps


def residual_bank(h: AssetHistory, st: AssetState, target_type: str, step_points: list[int],
                  cfg: dict[str, Any]) -> dict[int, np.ndarray]:
    ncfg, tcfg = cfg["numerical"], cfg["tails"]
    if h.is_transfer:
        return {}
    x = (return_steps(h.series) if target_type == "log_return" else level_steps(h.series)
         ).to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 600:
        return {}
    ew = ewma_var_path(x, float(ncfg["ewma_halflife"]))
    v300 = rolling_var(x, 299, 60)
    vlong = rolling_var(x, int(ncfg["long_window"]), 120)
    om = float(ncfg.get("lr_omega", 0.5))
    lr = (1 - om) * v300 + om * vlong
    cs = np.concatenate([[0.0], np.cumsum(x)])
    stride = int(tcfg.get("backtest_stride", 10))
    years = float(tcfg.get("backtest_years", 12))
    start = max(300, n - int(years * 252) - max(step_points))
    out: dict[int, np.ndarray] = {}
    phi = float(ncfg["vol_persistence"])
    scale = float(ncfg.get("scale", 1.0))
    for s in step_points:
        ts = np.arange(start, n - s, stride)
        if len(ts) < 20:
            continue
        V = aggregate_variance(ew[ts], lr[ts], float(s), phi) * st.variance_ratio.get(s, 1.0)
        V = V * scale * scale
        move = cs[ts + 1 + s] - cs[ts + 1]
        z = move / np.sqrt(np.maximum(V, 1e-24))
        z = z[np.isfinite(z)]
        out[int(s)] = z
    return out
