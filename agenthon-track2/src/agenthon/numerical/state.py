"""L3 — numerical state s_t: per-asset scale/drift/trend features and cross-asset dependence.

Everything here is a function of the unit's own panel at or before the as-of (point-in-time).
Features are vol-standardised (returns / trailing vol) so they transfer across assets and, for
F2, across currencies unseen in development.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..types import AssetHistory, AssetState, Card, Grid, NumericalState, RegimeBelief
from .regime import estimate_regime
from .statistical import ewma_var_path, variance_ratio
from .transform import level_steps, return_steps


def asset_steps(h: AssetHistory, target_type: str) -> pd.Series:
    s = h.series
    if target_type == "log_return" and h.asset_class in ("factor", "other"):
        return return_steps(s, log=True)
    if target_type == "log_return":
        return return_steps(s, log=True)
    return level_steps(s)


def _m0_window_steps(h: AssetHistory, target_type: str) -> np.ndarray:
    win = h.series.iloc[-300:]
    if target_type == "log_return":
        st = return_steps(win, log=False)  # M0 steps on the raw rows
    else:
        st = level_steps(win)
    return st.to_numpy(dtype=float)


def asset_state(h: AssetHistory, card: Card, step_counts: list[int], cfg: dict[str, Any]) -> AssetState:
    ncfg = cfg["numerical"]
    st = asset_steps(h, card.target_type)
    x = st.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    m0x = _m0_window_steps(h, card.target_type)
    anchor = 0.0 if card.target_type == "log_return" else float(h.series.iloc[-1])
    if len(x) < 3:
        x = np.array([0.0, 0.0, 0.0])
    sd_m0 = float(np.std(m0x, ddof=1)) if len(m0x) > 2 else float(np.std(x))
    drift_m0 = float(np.mean(m0x)) if len(m0x) else 0.0
    ew = ewma_var_path(x, float(ncfg["ewma_halflife"]))
    sd_ewma = float(np.sqrt(max(ew[-1], 0.0)))
    L = int(ncfg["long_window"])
    sd_long = float(np.std(x[-L:], ddof=1)) if len(x) > 20 else sd_m0
    drift_long = float(np.mean(x[-max(L, 2500):]))
    k = min(63, len(x))
    mom = float(np.sum(x[-k:]) / (max(sd_long, 1e-12) * np.sqrt(k)))
    vr = {int(s): variance_ratio(x, int(s), float(ncfg["vr_shrink_n0"]),
                                 tuple(ncfg["vr_clip"])) for s in step_counts}
    floor = None
    if h.asset_class == "rates" and card.target_type == "level" and ncfg.get("zlb_enabled", True):
        floor = float(ncfg.get("zlb_floor", -0.1))
    return AssetState(
        asset=h.asset, asset_class=h.asset_class, anchor=anchor,
        step_sd_m0=max(sd_m0, 1e-12), step_sd_ewma=max(sd_ewma, 1e-12),
        step_sd_long=max(sd_long, 1e-12), drift_m0=drift_m0, drift_long=drift_long,
        momentum_z=mom, vol_ratio=sd_ewma / max(sd_long, 1e-12), variance_ratio=vr,
        n_steps=int(len(x)), last_date=str(h.series.index[-1].date()), level_floor=floor,
        extras={"sd_m0_window": sd_m0},
    )


def step_correlation(histories: dict[str, AssetHistory], assets: list[str], target_type: str,
                     cfg: dict[str, Any]) -> np.ndarray:
    """Correlation of increments: blend of the M0-window daily correlation and a longer-window
    correlation of overlapping 5-step changes (robust to asynchronous closes across markets)."""
    n = len(assets)
    if n == 1:
        return np.ones((1, 1))
    frame = pd.DataFrame({a: asset_steps(histories[a], target_type) for a in assets}).dropna()
    if len(frame) < 30:
        return np.eye(n)
    daily = frame.iloc[-299:].corr().to_numpy()
    long = frame.iloc[-1260:]
    weekly = long.rolling(5).sum().dropna().corr().to_numpy() if len(long) > 60 else daily
    lam = float(cfg["numerical"].get("corr_shrink", 0.15))
    c = 0.5 * daily + 0.5 * weekly
    c = (1 - lam) * c + lam * daily  # extra weight on the recent regime
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    return nearest_correlation(c)


def nearest_correlation(c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Eigenvalue-clip to the PSD cone and rescale to unit diagonal (a valid correlation)."""
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    c2 = (v * np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.clip(np.diag(c2), eps, None))
    c2 = c2 / np.outer(d, d)
    np.fill_diagonal(c2, 1.0)
    return c2


def build_state(card: Card, grid: Grid, histories: dict[str, AssetHistory],
                cfg: dict[str, Any]) -> NumericalState:
    assets = grid.assets
    step_counts = sorted({c.steps for c in grid.cells})
    states = {a: asset_state(histories[a], card, step_counts, cfg) for a in assets}
    corr = step_correlation(histories, assets, card.target_type, cfg)
    regime = estimate_regime(states, cfg) if cfg["regime"].get("enabled", True) else \
        RegimeBelief(("normal",), np.array([1.0]), "global")
    return NumericalState(card=card, grid=grid, assets=states, step_corr=corr, regime=regime,
                          diagnostics={"step_counts": step_counts})
