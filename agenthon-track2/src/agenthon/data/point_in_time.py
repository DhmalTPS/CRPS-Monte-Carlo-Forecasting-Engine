"""L1 — the point-in-time data controller: X_<=t and T_<=t for one unit."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataError
from ..timeutil import intervals_days
from ..types import AssetHistory, Card, PointInTimeData
from .availability import asset_class, filter_series
from .panel_loader import asset_panel_map, read_panels
from .text_loader import load_corpus


def _detect_transfer(s: pd.Series) -> bool:
    """Early window + a lone as-of row: a gap much longer than the typical spacing before the end."""
    if len(s) < 5:
        return False
    gaps = intervals_days(s.index)
    med = float(np.median(gaps)) if len(gaps) else 1.0
    return bool(gaps[-1] > max(10 * med, 30) and (gaps[:-1] <= max(10 * med, 5)).mean() > 0.9)


def build_histories(panels: dict[str, pd.DataFrame], asof: str) -> tuple[dict[str, AssetHistory], int]:
    amap = asset_panel_map(panels)
    out: dict[str, AssetHistory] = {}
    dropped = 0
    for asset, stem in amap.items():
        df = panels[stem]
        sub = df[df["asset"] == asset]
        s = (sub.drop_duplicates("date", keep="last").set_index("date")["value"]
             .astype(float).sort_index())
        s = s[np.isfinite(s.to_numpy())]
        s, n = filter_series(s, asof)
        dropped += n
        if s.empty:
            continue
        panel_id = str(sub["panel"].iloc[0]) if len(sub) else stem
        out[asset] = AssetHistory(
            asset=asset, panel_id=panel_id, series=s,
            is_transfer=_detect_transfer(s), asset_class=asset_class(asset, panel_id + " " + stem),
        )
    return out, dropped


def point_in_time_filter(card: Card, panels_dir: pathlib.Path, text_dir: pathlib.Path, asof: str,
                         cfg: dict[str, Any]) -> PointInTimeData:
    panels = read_panels(panels_dir)
    histories, dropped = build_histories(panels, asof)
    missing = [a for a in card.assets if a not in histories]
    if missing:
        raise DataError(f"target asset(s) absent from every panel at or before {asof}: {missing}")
    tcfg = cfg.get("text", {})
    docs, dropped_docs = ([], 0)
    if tcfg.get("enabled", True):
        docs, dropped_docs = load_corpus(text_dir, asof, int(tcfg.get("max_docs", 60)),
                                         int(tcfg.get("max_chars_per_doc", 120000)))
    return PointInTimeData(asof=asof, histories=histories, docs=docs,
                           dropped_rows=dropped, dropped_docs=dropped_docs)
