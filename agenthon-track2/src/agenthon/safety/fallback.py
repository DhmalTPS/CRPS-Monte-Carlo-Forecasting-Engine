"""Fallback forecasts, ordered by quality: exact M0 (scores 1.0) -> minimal walk -> constant+noise."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from ..types import Card, Grid


def m0_fallback(card: Card, panels_dir: pathlib.Path) -> tuple[np.ndarray, Grid, str]:
    """Rebuild the context from scratch (the failure may have been anywhere) and emit M0's draws."""
    from ..data import build_histories, read_panels
    from ..numerical.random_walk import m0_forecast, minimal_walk, m0_seed
    from ..numerical.transform import build_grid

    hist, _ = build_histories(read_panels(panels_dir), card.asof)
    grid = build_grid(card, hist)
    try:
        f = m0_forecast(card, grid, hist, n_draws=500)
        n = max(500, card.n_draws_min)
        if n != 500:  # card floor above 500: tile M0's law with the same seed chain
            f = m0_forecast(card, grid, hist, n_draws=n)
        return f.draws, grid, "failsafe_m0"
    except Exception as exc:
        print(f"[agenthon] M0 fallback failed ({type(exc).__name__}); minimal walk", file=sys.stderr)
        n = max(500, card.n_draws_min)
        return minimal_walk(card, grid, hist, n, m0_seed(card.unit_id)), grid, "failsafe_minimal"


def constant_fallback(card: Card, anchors: dict[str, float] | None = None) -> tuple[np.ndarray, Grid]:
    """No panel at all: tiny-spread draws around 0/anchor. Admissible, scores badly, beats 4.0."""
    from ..types import Cell

    grid = Grid(tuple(Cell(a, int(h), int(h)) for a in sorted(card.assets) for h in sorted(card.horizons)))
    n = max(500, card.n_draws_min)
    rng = np.random.default_rng(12345)
    base = np.array([(anchors or {}).get(c.asset, 0.0) for c in grid.cells])
    scale = np.maximum(np.abs(base) * 0.05, 0.01)
    return base + rng.standard_normal((n, grid.d)) * scale, grid
