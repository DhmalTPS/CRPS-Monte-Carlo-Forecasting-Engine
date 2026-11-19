"""L2 — target transformation, step series, and horizon -> panel-step conversion.

Target semantics (README + docs/M0-BASELINE.md):
* ``level``      : forecast the level itself; the walk moves in first differences.
* ``log_return`` : panel rows are decimal per-step SIMPLE returns r; the target is the cumulative
                   log return sum(log(1 + r)) over the horizon, anchored at 0.

Gap rule (M0 section 3.3): a step spanning an interval longer than
max(10 x median spacing, 5 days) is dropped (missing, not zero). This is what stops a transfer
card's withheld decade from reading as one enormous step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import TransformError
from ..timeutil import intervals_days
from ..types import Card, Cell, Grid


def _gap_limit(dates: pd.DatetimeIndex) -> float:
    if len(dates) < 2:
        return 5.0
    iv = intervals_days(dates)
    return max(10.0 * float(np.median(iv)), 5.0)


def level_steps(s: pd.Series) -> pd.Series:
    """First differences dated at the row they end on, holes dropped."""
    if len(s) < 2:
        return pd.Series(dtype=float)
    lim = _gap_limit(pd.DatetimeIndex(s.index))
    iv = intervals_days(s.index)
    d = np.diff(s.to_numpy(dtype=float))
    out = pd.Series(d, index=s.index[1:])
    return out[iv <= lim]


def return_steps(s: pd.Series, log: bool = True) -> pd.Series:
    """Per-step returns (rows 1..n-1 that do not span a hole). log=True -> log(1+r)."""
    if len(s) < 2:
        return pd.Series(dtype=float)
    lim = _gap_limit(pd.DatetimeIndex(s.index))
    iv = intervals_days(s.index)
    r = s.to_numpy(dtype=float)[1:]
    if log:
        r = np.log1p(np.clip(r, -0.999999, None))
    out = pd.Series(r, index=s.index[1:])
    return out[iv <= lim]


def steps_for(s: pd.Series, target_type: str, log: bool = True) -> pd.Series:
    if target_type == "log_return":
        return return_steps(s, log=log)
    return level_steps(s)


def looks_like_returns(s: pd.Series) -> bool:
    """M0's tripwire: per-step returns have median |value| well below 0.2."""
    v = s.to_numpy(dtype=float)
    return bool(len(v) and np.median(np.abs(v[np.isfinite(v)])) < 0.2)


# --------------------------------------------------------------------------------------------
# Horizon -> steps (M0 section 3.7)
# --------------------------------------------------------------------------------------------
def _month_count(last: pd.Timestamp, target: pd.Timestamp) -> int:
    return 12 * (target.year - last.year) + (target.month - last.month)


def counted_steps(window_dates: pd.DatetimeIndex, last_obs: pd.Timestamp,
                  target: pd.Timestamp | None, declared: int, ratio_threshold: float = 2.0) -> int:
    """M0 step count for one cell. Falls back to the declared horizon exactly as M0 does."""
    if target is None or len(window_dates) < 3:
        return int(declared)
    iv = intervals_days(window_dates)
    lim = max(10.0 * float(np.median(iv)), 5.0)
    good = iv[iv <= lim]
    spacing = float(np.mean(good)) if len(good) else float(np.mean(iv))
    if spacing > 20.0:
        steps = _month_count(last_obs, target)
    else:
        steps = int(round((target - last_obs).days / spacing)) if spacing > 0 else 0
    if steps <= 0:
        return int(declared)
    ratio = max(steps, declared) / max(min(steps, declared), 1)
    return int(steps) if ratio >= ratio_threshold else int(declared)


def cell_targets(card: Card) -> dict[tuple[str, int], pd.Timestamp]:
    """Target dates when the card supplies them (target_dates or monthly observation periods).

    Both lists enumerate the grid in card order: for each asset, for each horizon (MONTHLY-HORIZONS
    specifies one observation period per (asset, horizon) row). A list of len(horizons) is applied
    to every asset.
    """
    out: dict[tuple[str, int], pd.Timestamp] = {}
    src = card.target_dates or card.observation_periods
    if not src:
        return out
    vals: list[pd.Timestamp] = []
    for x in src:
        try:
            vals.append(pd.Period(str(x)[:7], freq="M").to_timestamp() if len(str(x)) <= 7
                        else pd.Timestamp(str(x)[:10]))
        except (ValueError, TypeError):
            return {}
    pairs = [(a, h) for a in card.assets for h in card.horizons]
    if len(vals) == len(pairs):
        return dict(zip(pairs, vals))
    if len(vals) == len(card.horizons):
        return {(a, h): vals[j] for a in card.assets for j, h in enumerate(card.horizons)}
    return {}


def build_grid(card: Card, histories: dict, m0_window: int = 300,
               ratio_threshold: float = 2.0) -> Grid:
    """The grid in M0 order with per-cell step counts."""
    targets = cell_targets(card)
    cells: list[Cell] = []
    for a in sorted(card.assets):
        h = histories.get(a)
        if h is None:
            raise TransformError(f"no history for target {a}")
        s = h.series
        win = s.iloc[-m0_window:]
        last_obs = pd.Timestamp(s.index[-1])
        for hz in sorted(card.horizons):
            st = counted_steps(pd.DatetimeIndex(win.index), last_obs, targets.get((a, hz)), hz,
                               ratio_threshold)
            cells.append(Cell(asset=a, horizon=int(hz), steps=max(int(st), 1)))
    return Grid(cells=tuple(cells))
