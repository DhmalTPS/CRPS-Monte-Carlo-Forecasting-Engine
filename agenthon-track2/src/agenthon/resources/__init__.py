"""Vintaged offline artifacts (fitted hyper-parameters, calibration maps, policy weights).

Artifact policy: "Features, labels and derived material used for fitting, adaptation, model
selection and calibration must have been available by each relevant task cutoff." Every artifact
therefore carries `fit_cutoff`, the last date of data it was fitted on, and `select_vintage` returns
the newest vintage with fit_cutoff < card as-of. A card older than every vintage uses no fitted
artifact at all (built-in literature priors). This makes the practice board and the sealed set
equally compliant.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

_DIR = pathlib.Path(__file__).parent


def load_artifact(name: str) -> dict[str, Any]:
    p = _DIR / name
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def select_vintage(artifact: dict[str, Any], asof: str) -> dict[str, Any] | None:
    best = None
    for v in artifact.get("vintages", []):
        fc = str(v.get("fit_cutoff", "9999"))
        if fc < asof and (best is None or fc > str(best.get("fit_cutoff"))):
            best = v
    return best


def _strip_private(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def vintage_overrides(asof: str, family: str | None = None) -> tuple[dict[str, Any], str]:
    """Config overrides from params_vintages.json valid at `asof` (+ that vintage's family block)."""
    v = select_vintage(load_artifact("params_vintages.json"), asof)
    if v is None:
        return {}, "priors"
    try:
        from ..config import _deep_merge

        ov = dict(v.get("overrides", {}))
        fam = (v.get("family_overrides") or {}).get(family or "")
        label = f"vintage<= {v.get('fit_cutoff')}"
        if fam:
            ov = _deep_merge(ov, _strip_private(fam))
            label += f" +{family}"
        return ov, label
    except Exception:
        print("[agenthon] malformed parameter vintage ignored", file=sys.stderr)
        return {}, "priors"
