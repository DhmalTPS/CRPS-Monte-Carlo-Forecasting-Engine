"""L5-L7 — assemble the text-blind prior P0 as a path-coherent `JointBelief`.

Tiers (docs Section 8.6, simplified per Correction C3):
  Tier 0  M0 replica (random walk)             -> the fail-safe and the normalisation reference
  Tier 1  vol term-structure model (statistical.py), macro trend model, transfer model
  Tier 1b historical analogue vote (conditional, off unless validated)
  Tier 2  foundation model (disabled: pretrained checkpoints need organizer approval)

"Ensembling" is done where it is statistically meaningful for a scale-location forecast: the
variance forecast is a log-space blend of estimators (EWMA / M0 window / long run) under a
mean-reverting term structure, and the centre is a shrunk blend of drift estimators. Averaging
full distributions of near-identical Gaussian walks would only add noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..timeutil import intervals_days
from ..types import AssetHistory, Card, Grid, JointBelief, NumericalState, Scenario
from . import analogue as analogue_mod
from .statistical import macro_drift_and_sd, transfer_relative_vol, vol_term_structure
from .transform import level_steps


def _is_monthly(h: AssetHistory) -> bool:
    iv = intervals_days(h.series.index[-60:])
    return bool(len(iv) and float(np.median(iv)) > 20.0)


def _proxy_rel_vols(histories: dict[str, AssetHistory], exclude: str) -> list[float]:
    out = []
    for a, h in histories.items():
        if a == exclude or h.asset_class != "fx" or len(h.series) < 300:
            continue
        v = h.series.to_numpy(dtype=float)[-300:]
        v = v[v > 0]
        if len(v) > 50:
            out.append(float(np.std(np.diff(np.log(v)), ddof=1)))
    return out


def asset_variance_path(asset: str, state: NumericalState, histories: dict[str, AssetHistory],
                        step_points: list[int], cfg: dict[str, Any],
                        scale_mult: float = 1.0) -> tuple[np.ndarray, float, dict[str, Any]]:
    """(cumulative variance at each step point, per-step drift, diagnostics) for one asset."""
    ncfg = cfg["numerical"]
    st = state.assets[asset]
    h = histories[asset]
    card = state.card
    diag: dict[str, Any] = {}
    if h.is_transfer:
        early = h.series.iloc[:-1]
        rel = np.diff(np.log(np.clip(early.to_numpy(dtype=float), 1e-12, None)))
        # gap-drop inside the early window too
        rel = rel[intervals_days(early.index) <= max(10 * np.median(intervals_days(early.index)), 5)]
        rv = transfer_relative_vol(rel, _proxy_rel_vols(histories, asset),
                                   float(ncfg.get("transfer_proxy_weight", 0.5)))
        sd_step = rv * abs(st.anchor)
        var = np.array([s * sd_step ** 2 for s in step_points]) * scale_mult ** 2
        diag.update(model="transfer", rel_vol=rv)
        return var, 0.0, diag
    if card.target_type == "level" and _is_monthly(h):
        dx = level_steps(h.series).to_numpy(dtype=float)
        drift, sd, sd_mu = macro_drift_and_sd(dx, float(ncfg.get("macro_trend_halflife", 6.0)),
                                              float(ncfg.get("macro_trend_weight", 0.7)))
        var = np.array([s * sd ** 2 + (s ** 2) * sd_mu ** 2 for s in step_points]) * scale_mult ** 2
        diag.update(model="macro_trend", drift=drift, sd=sd, sd_mu=sd_mu)
        return np.maximum.accumulate(var), drift, diag
    omega = float(ncfg.get("lr_omega", 0.5))
    sig_lr = float(np.sqrt((1 - omega) * st.step_sd_m0 ** 2 + omega * st.step_sd_long ** 2))
    sig_now = st.step_sd_ewma
    var = vol_term_structure(sig_now, sig_lr, step_points, float(ncfg["vol_persistence"]),
                             st.variance_ratio, scale=float(ncfg.get("scale", 1.0)) * scale_mult)
    if card.target_type == "log_return":
        k = float(ncfg.get("drift_shrink_return", 0.25))
        drift = k * st.drift_m0 + (1 - k) * st.drift_long
    else:
        drift = float(ncfg.get("drift_shrink", 0.0)) * st.drift_m0
    diag.update(model="vol_term_structure", sig_now=sig_now, sig_lr=sig_lr)
    return var, drift, diag


def numerical_forecast(state: NumericalState, histories: dict[str, AssetHistory],
                       cfg: dict[str, Any], scale_bias: dict[str, float] | None = None,
                       df: float | None = None) -> JointBelief:
    grid: Grid = state.grid
    card: Card = state.card
    step_points = sorted({c.steps for c in grid.cells})
    var_path: dict[str, np.ndarray] = {}
    drift: dict[str, float] = {}
    prov: dict[str, Any] = {"assets": {}}
    for a in grid.assets:
        mult = float((scale_bias or {}).get(a, 1.0))
        v, d, diag = asset_variance_path(a, state, histories, step_points, cfg, mult)
        if cfg["analogue"].get("enabled", False):
            am = analogue_mod.analogue_scale(histories[a], card.target_type, step_points, cfg)
            if am is not None:
                w = float(cfg["analogue"].get("weight", 0.15))
                v = v * np.exp(w * np.log(np.clip(am, 0.5, 2.0)) * 2.0)
                diag["analogue_scale"] = am.tolist()
        var_path[a] = np.maximum.accumulate(np.maximum(v, 1e-18))
        drift[a] = d
        prov["assets"][a] = diag
    mu = np.array([state.assets[c.asset].anchor + c.steps * drift[c.asset] for c in grid.cells])
    floors = {a: s.level_floor for a, s in state.assets.items() if s.level_floor is not None}
    base = Scenario("base", 1.0, np.zeros(grid.d), np.ones(grid.d), "numerical prior")
    return JointBelief(grid=grid, mu=mu, step_points=tuple(step_points), var_path=var_path,
                       corr=state.step_corr.copy(),
                       df=float(df if df is not None else cfg["numerical"].get("df", 7.0)),
                       scenarios=[base], floors=floors, provenance=prov)
