"""Local mirror of the admissibility gates g0-g3 (refuse to emit anything that would fail them).

g0 exactly three regular files; g1 sidecar schema (unit_id, asof, representation='samples',
asset_ids, horizons, integer n_draws in [200, 20000]) and a non-blank rationale; g2 unit_id/asof
bound to the trusted card; g3 declared grid == card grid, every (draw, asset, horizon) exactly
once, draw ids 0..n-1, finite values.
"""

from __future__ import annotations

import re

import numpy as np

from ..errors import OutputValidationError
from ..types import Card, Grid

MIN_DRAWS, MAX_DRAWS = 200, 20000
FILES = ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_meta(card: Card, n_draws: int, method: str, seed: int) -> dict:
    return {
        "unit_id": card.unit_id,
        "asof": card.asof,
        "representation": "samples",
        "asset_ids": list(card.assets),
        "horizons": [int(h) for h in card.horizons],
        "n_draws": int(n_draws),
        "target": card.target_type if card.target_type in ("level", "log_return") else "level",
        "rationale": {"file": "forecast_rationale.md", "method": method[:200]},
    }


def validate_forecast(draws: np.ndarray, grid: Grid, card: Card, meta: dict, rationale: str) -> None:
    if draws.ndim != 2 or draws.shape[1] != grid.d:
        raise OutputValidationError(f"draw matrix shape {draws.shape} != [n, {grid.d}]")
    n = draws.shape[0]
    if not (max(MIN_DRAWS, card.n_draws_min) <= n <= MAX_DRAWS):
        raise OutputValidationError(f"n_draws {n} outside [{max(MIN_DRAWS, card.n_draws_min)}, {MAX_DRAWS}]")
    if not np.all(np.isfinite(draws)):
        raise OutputValidationError("non-finite draw values")
    if np.any(draws.std(axis=0) <= 0):
        raise OutputValidationError("a cell has zero spread (point forecast dressed as a distribution)")
    if {(c.asset, c.horizon) for c in grid.cells} != {(a, int(h)) for a in card.assets for h in card.horizons}:
        raise OutputValidationError("grid does not match the card")
    if meta.get("unit_id") != card.unit_id or not _ISO.match(str(meta.get("asof", ""))) \
            or meta.get("asof") != card.asof:
        raise OutputValidationError("sidecar unit_id/asof not bound to the card")
    if meta.get("n_draws") != n or meta.get("representation") != "samples":
        raise OutputValidationError("sidecar n_draws/representation inconsistent")
    if not rationale or not rationale.strip():
        raise OutputValidationError("empty rationale")
