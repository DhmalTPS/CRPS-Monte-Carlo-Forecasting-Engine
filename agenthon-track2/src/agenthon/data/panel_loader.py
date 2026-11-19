"""L1 — read every parquet panel under --panels (or its parent, the exemplar layout).

Asset resolution follows M0 exactly (docs/M0-BASELINE.md section 3.1): an asset's history comes
from the *first* parquet, in sorted filename order, that contains a row for it. Both `asset` and
`asset_id` column spellings are accepted.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from ..errors import DataError

ASSET_COLS = ("asset", "asset_id")


def panel_files(panels_dir: pathlib.Path) -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for d in (panels_dir, panels_dir.parent):
        try:
            found = sorted(p for p in d.glob("*.parquet") if p.is_file())
        except OSError:
            found = []
        if found:
            break
    return found


def _normalise(df: pd.DataFrame, panel_name: str) -> pd.DataFrame | None:
    col = next((c for c in ASSET_COLS if c in df.columns), None)
    if col is None or "date" not in df.columns or "value" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="coerce"),
            "asset": df[col].astype(str),
            "value": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    out["panel"] = str(df["panel_id"].iloc[0]) if "panel_id" in df.columns and len(df) else panel_name
    return out.dropna(subset=["date"])


def read_panels(panels_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """filename stem -> long frame [date, asset, value, panel]; unreadable files are skipped."""
    files = panel_files(panels_dir)
    if not files:
        raise DataError(f"no .parquet found under {panels_dir} (or its parent)")
    out: dict[str, pd.DataFrame] = {}
    for p in files:
        try:
            df = _normalise(pd.read_parquet(p), p.stem)
        except Exception as exc:  # a corrupt context panel must not kill the card
            print(f"[agenthon] panel {p.name} unreadable ({type(exc).__name__}); skipped",
                  file=sys.stderr)
            continue
        if df is not None and len(df):
            out[p.stem] = df
    if not out:
        raise DataError("every panel file was unreadable")
    return out


def asset_panel_map(panels: dict[str, pd.DataFrame]) -> dict[str, str]:
    """asset -> stem of the first panel (sorted filename order) carrying it. M0 rule."""
    m: dict[str, str] = {}
    for stem in sorted(panels):
        for a in panels[stem]["asset"].unique():
            m.setdefault(str(a), stem)
    return m
