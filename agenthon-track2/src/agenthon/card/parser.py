"""L0 — read card.toml (+ forecast_spec.json, forecast_card.md) into a `Card`.

Where the card lives: the harness mounts the *unit directory* at /input, so the card is at
/input/card.toml while panels are /input/panels/. We look in --panels, its parent, then /input.

Trusted as-of = [provenance] data_cutoff (what g2 binds forecast_meta.json against); the CLI's
--asof is cross-checked by the validator and the information set is min(--asof, data_cutoff).
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from typing import Any

from ..errors import CardError
from ..types import FAMILIES, Card


def find_card(panels_dir: pathlib.Path | None, explicit: pathlib.Path | None = None) -> pathlib.Path:
    cands: list[pathlib.Path] = []
    if explicit is not None:
        cands.append(explicit)
    if panels_dir is not None:
        cands += [panels_dir / "card.toml", panels_dir.parent / "card.toml"]
    cands.append(pathlib.Path("/input/card.toml"))
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    raise CardError(f"card.toml not found (looked in: {', '.join(str(c) for c in cands)})")


def _family(raw: dict[str, Any]) -> str:
    fam = str(raw.get("metadata", {}).get("category", "") or "").upper()
    if fam in FAMILIES:
        return fam
    tid = str(raw.get("task", {}).get("id", "")).upper()
    for f in FAMILIES:
        if f"-{f.split('-')[1]}-" in tid:  # e.g. "t2-F4-..." -> T2-F4
            return f
    return "T2-UNKNOWN"


def _read_text(path: pathlib.Path, limit: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit] if path.is_file() else ""
    except OSError:
        return ""


def parse_card(card_path: pathlib.Path) -> Card:
    try:
        raw = tomllib.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CardError(f"card.toml unreadable: {type(exc).__name__}") from None
    try:
        task = raw["task"]
        tgt = raw["targets"]
        unit_id = str(task["id"])
        assets = tuple(str(a) for a in tgt["asset_ids"])
        horizons = tuple(int(h) for h in tgt["horizons"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CardError(f"card.toml lacks a usable [task]/[targets]: {exc}") from None
    prov = raw.get("provenance", {})
    asof = str(prov.get("data_cutoff") or raw.get("text", {}).get("cutoff") or "")[:10]
    params = raw.get("scoring", {}).get("params", {})
    w = params.get("weights", {}) or {}
    weights = (float(w.get("marginal", 0.5)), float(w.get("joint", 0.3)), float(w.get("tail", 0.2)))
    meta = raw.get("metadata", {})
    spec_path = card_path.parent / "forecast_spec.json"
    spec: dict[str, Any] = {}
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                spec = {}
        except (ValueError, OSError):
            spec = {}
    st = spec.get("targets", {}) if isinstance(spec.get("targets"), dict) else {}
    obs = st.get("observation_periods") or tgt.get("observation_periods")
    tdates = tgt.get("target_dates") or st.get("target_dates")
    return Card(
        unit_id=unit_id,
        asof=asof,
        family=_family(raw),
        assets=assets,
        horizons=horizons,
        target_type=str(tgt.get("target_type", meta.get("target_type", "level"))),
        target_frequency=str(tgt.get("target_frequency", meta.get("target_frequency", "daily"))),
        value_unit=str(tgt.get("value_unit", "")),
        n_draws_min=int(params.get("n_draws_min", 200) or 200),
        weights=weights,
        tail_levels=tuple(float(x) for x in params.get("tail_levels", (0.01, 0.05, 0.95, 0.99))),
        joint=str(params.get("joint", "variogram")),
        title=str(task.get("title", "")),
        description=str(meta.get("description", "")),
        text_notes=str(raw.get("text", {}).get("notes", "")),
        prose=_read_text(card_path.parent / "forecast_card.md"),
        observation_periods=tuple(str(x) for x in obs) if obs else None,
        target_dates=tuple(str(x)[:10] for x in tdates) if tdates else None,
        card_dir=str(card_path.parent),
        raw=raw,
    )
