"""L23 — write forecast.parquet, forecast_meta.json, forecast_rationale.md (and nothing else).

Atomic: each file is written under a temporary name in the output directory and os.replace()d, so
a crash can never leave a half-written parquet that parses. Temporary names are removed by
`cleanup_temporaries`, which the fail-safe always calls (g0 wants exactly three files).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..types import Card, Grid
from .schema import FILES, validate_forecast

_TMP = ".agenthon-tmp-"


def to_table(draws: np.ndarray, grid: Grid) -> pa.Table:
    n, d = draws.shape
    return pa.table({
        "draw": pa.array(np.repeat(np.arange(n, dtype=np.int32), d), type=pa.int32()),
        "asset": pa.array([c.asset for c in grid.cells] * n, type=pa.string()),
        "horizon": pa.array(np.tile(np.array([c.horizon for c in grid.cells], dtype=np.int32), n),
                            type=pa.int32()),
        "value": pa.array(draws.reshape(-1).astype(np.float64), type=pa.float64()),
    })


def _atomic_write(path: pathlib.Path, writer) -> None:
    tmp = path.with_name(_TMP + path.name)
    writer(tmp)
    os.replace(tmp, path)


def cleanup_temporaries(out_dir: pathlib.Path) -> None:
    try:
        for p in out_dir.iterdir():
            if p.name.startswith(_TMP):
                p.unlink(missing_ok=True)
    except OSError:
        pass


def write_forecast(out_path: pathlib.Path, draws: np.ndarray, grid: Grid, card: Card,
                   meta: dict, rationale: str) -> None:
    validate_forecast(draws, grid, card, meta, rationale)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILES[0] if out_path.name != FILES[0] else out_path
    table = to_table(draws, grid)
    _atomic_write(target, lambda p: pq.write_table(table, p, compression="zstd"))
    _atomic_write(out_dir / FILES[1], lambda p: p.write_text(json.dumps(meta, indent=2) + "\n",
                                                            encoding="utf-8"))
    _atomic_write(out_dir / FILES[2], lambda p: p.write_text(rationale, encoding="utf-8"))
    cleanup_temporaries(out_dir)
