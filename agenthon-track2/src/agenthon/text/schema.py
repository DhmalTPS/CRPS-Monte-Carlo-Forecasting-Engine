"""L13 output schema for Z_t and a tolerant coercer (open house model => budget for parse failure).

Coercion is lenient on form (strings for numbers, synonyms for enum values) and strict on
meaning: anything that cannot be mapped becomes the NEUTRAL value, so a malformed field can only
reduce the text's influence, never invent one.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import LLMParseError
from ..types import AssetSignal, ScenarioView, SemanticSignal

MAGNITUDES = ("none", "weak", "moderate", "strong", "extreme")
VOL_VIEWS = ("lower", "same", "higher", "much_higher")
TAILS = ("none", "up", "down", "both")
PERSIST = ("transient", "medium", "persistent")
REGIME = ("supports_calm", "neutral", "supports_stress")

SCHEMA_EXAMPLE = {
    "assets": {"<ASSET_ID>": {"p_up": 0.5, "magnitude": "weak", "vol_view": "same",
                              "tail_side": "none"}},
    "scenarios": [{"name": "base", "probability": 0.7, "direction": {"<ASSET_ID>": 0},
                   "magnitude": "weak", "vol_view": "same"}],
    "persistence": "medium",
    "regime_evidence": "neutral",
    "confidence": 0.3,
    "novelty": 0.3,
    "established": False,
    "cited_docs": ["<doc_id>"],
    "summary": "<= 60 words",
}

_SYN = {"high": "higher", "elevated": "higher", "very_high": "much_higher", "much higher": "much_higher",
        "low": "lower", "unchanged": "same", "normal": "same", "medium": "moderate",
        "large": "strong", "small": "weak", "severe": "extreme", "neutral": "none"}


def extract_json(text: str) -> dict[str, Any]:
    if not text:
        raise LLMParseError("empty completion")
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M)
    start = t.find("{")
    if start < 0:
        raise LLMParseError("no JSON object")
    depth, end = 0, -1
    in_str, esc = False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise LLMParseError("unterminated JSON object")
    raw = t[start:end + 1]
    try:
        obj = json.loads(raw)
    except ValueError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))  # trailing commas
        except ValueError as exc:
            raise LLMParseError(f"invalid JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise LLMParseError("top level is not an object")
    return obj


def _enum(v: Any, allowed: tuple[str, ...], default: str) -> str:
    s = str(v or "").strip().lower().replace("-", "_")
    s = _SYN.get(s, s)
    return s if s in allowed else default


def _prob(v: Any, default: float = 0.5) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x < 0.0 or x > 100.0:
        return default  # out-of-domain garbage is unusable, never clipped to an extreme
    if x > 1.0:
        x /= 100.0  # a percentage
    return float(x)


def coerce_signal(obj: dict[str, Any], assets: list[str], doc_ids: set[str]) -> SemanticSignal:
    av = obj.get("assets") if isinstance(obj.get("assets"), dict) else {}
    sigs: dict[str, AssetSignal] = {}
    for a in assets:
        row = av.get(a) or av.get(a.upper()) or av.get(a.lower()) or {}
        if not isinstance(row, dict):
            row = {}
        p_up = _prob(row.get("p_up"), 0.5)
        if "direction" in row and "p_up" not in row:
            d = str(row.get("direction", "")).lower()
            p_up = 0.65 if d in ("up", "higher", "+1", "1") else 0.35 if d in ("down", "lower", "-1") else 0.5
        sigs[a] = AssetSignal(a, p_up, 1.0 - p_up, _enum(row.get("magnitude"), MAGNITUDES, "none"),
                              _enum(row.get("vol_view"), VOL_VIEWS, "same"),
                              _enum(row.get("tail_side"), TAILS, "none"))
    scen: list[ScenarioView] = []
    for s in obj.get("scenarios", []) if isinstance(obj.get("scenarios"), list) else []:
        if not isinstance(s, dict):
            continue
        dirs = {}
        dd = s.get("direction") if isinstance(s.get("direction"), dict) else {}
        for a in assets:
            try:
                dirs[a] = int(max(-1, min(1, round(float(dd.get(a, 0))))))
            except (TypeError, ValueError):
                dirs[a] = 0
        scen.append(ScenarioView(str(s.get("name", "scenario"))[:40], _prob(s.get("probability"), 0.0),
                                 dirs, _enum(s.get("magnitude"), MAGNITUDES[1:], "weak"),
                                 _enum(s.get("vol_view"), VOL_VIEWS, "same")))
    tot = sum(s.probability for s in scen)
    if scen and tot > 0:
        for s in scen:
            s.probability /= tot
    scen = scen[:4]
    cited = tuple(str(c) for c in obj.get("cited_docs", []) if str(c) in doc_ids) \
        if isinstance(obj.get("cited_docs"), list) else ()
    return SemanticSignal(
        assets=sigs, scenarios=scen,
        persistence=_enum(obj.get("persistence"), PERSIST, "transient"),
        regime_evidence=_enum(obj.get("regime_evidence"), REGIME, "neutral"),
        confidence=_prob(obj.get("confidence"), 0.0),
        novelty=_prob(obj.get("novelty"), 0.0),
        # strict: only a JSON boolean true (or the literal string "true") counts as established,
        # because "established" is the one flag that permits text-driven narrowing
        established=obj.get("established") is True or str(obj.get("established")).strip().lower() == "true",
        source="llm", cited_docs=cited,
        summary=str(obj.get("summary", ""))[:600],
    )
