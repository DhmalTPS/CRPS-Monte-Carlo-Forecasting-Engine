"""Prompt loading: repo prompts/ (dev), /opt/agenthon/prompts (image), embedded fallback."""

from __future__ import annotations

import os
import pathlib

_FALLBACK_SYSTEM = (
    "You are a careful macro-financial analyst. Use ONLY the dated documents given (all dated on "
    "or before the as-of); never use knowledge of later events. Never state a market number; use "
    "magnitude classes. Financial text is low signal: most documents justify p_up 0.4-0.6. Keep "
    "disagreement (lower confidence, higher vol) rather than averaging it away. Answer with ONE "
    "JSON object and nothing else."
)


def _dirs() -> list[pathlib.Path]:
    out = []
    if os.environ.get("AGENTHON_PROMPT_DIR"):
        out.append(pathlib.Path(os.environ["AGENTHON_PROMPT_DIR"]))
    out.append(pathlib.Path(__file__).resolve().parents[3] / "prompts")
    out.append(pathlib.Path("/opt/agenthon/prompts"))
    return out


def load_prompt(name: str) -> str:
    for d in _dirs():
        p = d / name
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except OSError:
            continue
    if name == "system.txt":
        return _FALLBACK_SYSTEM
    return ("AS-OF {asof}. CARD {title}. NOTES {notes}. TARGETS {conventions}. HORIZONS {horizons}. "
            "STATE {numeric_state}. DOCUMENTS {evidence}. Return one JSON object: {schema}")
