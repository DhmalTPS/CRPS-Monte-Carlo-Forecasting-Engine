"""L13/L14 — evidence -> Z_t (LLM, schema-constrained) with a lexicon fallback.

The mapping Z -> numbers is NOT done here: it happens in fusion/distribution.py through the
panel's own empirical |z| quantiles (L14 semantic-to-empirical map). Here we only produce Z.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from ..errors import AgenthonError, LLMParseError
from ..types import Card, Event, NumericalState, SemanticSignal, AssetSignal
from .lexicon import USD_PER_UNIT
from .llm import HouseClient, log
from .prompts import load_prompt
from .schema import SCHEMA_EXAMPLE, coerce_signal, extract_json


def convention(asset: str, asset_class: str) -> str:
    a = asset.upper()
    if asset_class == "rates":
        return f"{a}: US Treasury yield in % per annum; up = yield rises"
    if asset_class in ("fx", "em_fx"):
        if a in USD_PER_UNIT:
            return f"{a}: US dollars per 1 {a}; up = {a} strengthens against USD"
        return f"{a}: {a} per 1 US dollar; up = {a} WEAKENS against USD (USD strengthens)"
    if asset_class == "factor":
        return f"{a}: cumulative log return of the {a} factor portfolio over the horizon; up = positive"
    if asset_class == "macro":
        return f"{a}: macro level (monthly); up = above the numeric model's trend path"
    return f"{a}: level; up = higher"


def _numeric_state_text(state: NumericalState) -> str:
    lines = []
    for a, s in state.assets.items():
        lines.append(f"- {a}: last value {s.anchor:.6g} on {s.last_date}; 3-month trend z={s.momentum_z:+.2f}; "
                     f"current/long-run volatility ratio {s.vol_ratio:.2f}")
    reg = ", ".join(f"{l} {p:.2f}" for l, p in zip(state.regime.labels, state.regime.probs))
    lines.append(f"- volatility regime probabilities: {reg}")
    return "\n".join(lines)


def _evidence_text(events: list[Event]) -> str:
    return "\n\n".join(f"[{e.doc_id} | {e.timestamp} | {e.doc_type}{' | ' + e.institution if e.institution else ''}]\n{e.text}"
                       for e in events)


def build_messages(card: Card, state: NumericalState, events: list[Event]) -> list[dict[str, str]]:
    conv = "\n".join(convention(a, state.assets[a].asset_class) for a in state.grid.assets)
    schema = json.dumps({**SCHEMA_EXAMPLE,
                         "assets": {a: SCHEMA_EXAMPLE["assets"]["<ASSET_ID>"] for a in state.grid.assets}})
    user = load_prompt("structured_reasoning.txt").format(
        asof=card.asof, title=card.title,
        notes=(card.description + "\n" + card.text_notes).strip()[:1500] or "(none)",
        conventions=conv, horizons=sorted({c.horizon for c in state.grid.cells}),
        numeric_state=_numeric_state_text(state), evidence=_evidence_text(events) or "(no documents)",
        schema=schema)
    return [{"role": "system", "content": load_prompt("system.txt")},
            {"role": "user", "content": user}]


def llm_signal(client: HouseClient, card: Card, state: NumericalState, events: list[Event],
               deadline: float, cfg: dict[str, Any]) -> SemanticSignal | None:
    msgs = build_messages(card, state, events)
    doc_ids = {e.doc_id for e in events}
    retries = int(cfg["llm"].get("retries", 1))
    for attempt in range(retries + 1):
        try:
            out = client.chat(msgs, deadline=deadline)
            sig = coerce_signal(extract_json(out), state.grid.assets, doc_ids)
            return sig
        except LLMParseError as exc:
            log(f"parse failure (attempt {attempt + 1}): {exc}")
            msgs = msgs + [{"role": "user", "content": "Your answer was not a single valid JSON object. "
                            "Reply again with ONLY the JSON object, no other text."}]
        except AgenthonError as exc:
            log(f"unavailable: {exc}")
            return None
        if time.monotonic() > deadline - 15:
            return None
    return None


def lexicon_signal(state: NumericalState, events: list[Event]) -> SemanticSignal:
    """Weak, numeric-only fallback: widen on stress vocabulary, tilt rates only on Fed tone.

    Deliberately low confidence; the gate additionally caps any lexicon-sourced signal.
    """
    assets = state.grid.assets
    sig = SemanticSignal.neutral(assets, source="lexicon")
    if not events:
        return sig
    stress = float(np.mean([e.stress for e in events]))
    tails = float(np.mean([min(e.tail_terms, 5) for e in events]))
    fed = [e.hawk for e in events if "Federal Reserve" in (e.institution or "") or e.doc_type.startswith("fomc")]
    tone = float(np.mean(fed)) if fed else 0.0
    vol = "much_higher" if stress > 0.6 else "higher" if stress > 0.35 else "same"
    for a in assets:
        cls = state.assets[a].asset_class
        p_up = 0.5 + (0.08 * float(np.clip(tone, -1, 1)) if cls == "rates" else 0.0)
        mag = "weak" if abs(p_up - 0.5) > 0.03 else "none"
        sig.assets[a] = AssetSignal(a, p_up, 1 - p_up, mag, vol, "both" if tails > 1.0 else "none")
    sig.regime_evidence = "supports_stress" if stress > 0.45 else "neutral"
    sig.confidence = float(np.clip(0.15 + 0.2 * abs(tone) + 0.2 * stress, 0.0, 0.4))
    sig.novelty = 0.3
    return sig
