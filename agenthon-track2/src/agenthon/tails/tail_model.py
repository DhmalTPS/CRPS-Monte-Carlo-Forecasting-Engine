"""L20 — combine the residual bank into per-asset scale bias, a card-level dof, and |z| quantiles
used by the semantic-to-numeric magnitude map (L14)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import AssetHistory, NumericalState
from .residuals import residual_bank
from .shrinkage import shrink_df, shrink_log_scale

_STD_ABS_Q = {"weak": 0.674, "moderate": 1.150, "strong": 1.645, "extreme": 2.326}


@dataclass
class TailProfile:
    scale_bias: dict[str, float] = field(default_factory=dict)
    df: float = 7.0
    abs_z_quantiles: dict[str, float] = field(default_factory=lambda: dict(_STD_ABS_Q))
    weights: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0


def tail_profile(state: NumericalState, histories: dict[str, AssetHistory], cfg: dict[str, Any],
                 prior_df: float) -> TailProfile:
    tcfg = cfg["tails"]
    prof = TailProfile(df=prior_df)
    if not tcfg.get("in_unit_backtest", True):
        return prof
    step_points = sorted({c.steps for c in state.grid.cells})
    stride = int(tcfg.get("backtest_stride", 10))
    all_z: list[np.ndarray] = []
    dfs: list[float] = []
    for a in state.grid.assets:
        bank = residual_bank(histories[a], state.assets[a], state.card.target_type, step_points, cfg)
        if not bank:
            continue
        biases, ws = [], []
        for s, z in bank.items():
            b, w = shrink_log_scale(z, stride, s, float(tcfg["n0_scale"]),
                                    tuple(tcfg["scale_bias_clip"]))
            biases.append(np.log(b))
            ws.append(w)
            dfs.append(shrink_df(z, stride, s, prior_df, float(tcfg["n0_df"]),
                                 tuple(tcfg["df_clip"])))
            all_z.append(z)
        if biases:
            prof.scale_bias[a] = float(np.exp(np.mean(biases)))
            prof.weights[a] = float(np.mean(ws))
    if dfs:
        prof.df = float(np.exp(np.mean(np.log(dfs))))
    if all_z:
        z = np.abs(np.concatenate(all_z))
        z = z / max(np.sqrt(np.mean(np.clip(z, 0, 8) ** 2)), 1e-9)
        prof.n_obs = int(len(z))
        if len(z) >= 200:
            emp = {k: float(np.quantile(z, p)) for k, p in
                   (("weak", 0.5), ("moderate", 0.75), ("strong", 0.9), ("extreme", 0.98))}
            prof.abs_z_quantiles = {k: 0.5 * emp[k] + 0.5 * _STD_ABS_Q[k] for k in emp}
    return prof
