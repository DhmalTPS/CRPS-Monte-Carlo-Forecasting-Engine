"""L22 — apply the vintaged PIT maps (asset class x horizon bucket) to the belief draws.

Order matters: calibration acts on the HONEST belief; the score-aware decision layer acts after
it. On any problem the draws pass through unchanged (uncalibrated is the documented fallback).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..resources import load_artifact, select_vintage
from ..types import Grid
from .maps import apply_inverse_pit


def horizon_bucket(steps: int) -> str:
    return "h<=10" if steps <= 10 else "h<=40" if steps <= 40 else "h<=100" if steps <= 100 else "h>100"


def calibrate(draws: np.ndarray, grid: Grid, classes: dict[str, str], asof: str,
              cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    if not cfg.get("calibration", {}).get("enabled", True):
        return draws, "disabled"
    try:
        v = select_vintage(load_artifact("calibration_vintages.json"), asof)
        if v is None:
            return draws, "no vintage"
        maps = v.get("maps", {})
        out = draws
        used = []
        for a in grid.assets:
            by_bucket: dict[str, list[int]] = {}
            for i in grid.cells_of(a):
                by_bucket.setdefault(horizon_bucket(grid.cells[i].steps), []).append(i)
            for b, cells in by_bucket.items():
                knots = maps.get(f"{classes.get(a, 'other')}|{b}")
                if knots:
                    out = apply_inverse_pit(out, cells, knots)
                    used.append(f"{a}:{b}")
        return out, f"vintage<= {v.get('fit_cutoff')} ({len(used)} maps)"
    except Exception as exc:
        print(f"[agenthon] calibration skipped: {type(exc).__name__}", file=sys.stderr)
        return draws, "error"
