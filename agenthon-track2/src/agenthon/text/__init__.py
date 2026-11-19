"""Text branch L8 -> L14: T_<=t -> events -> dedup -> query -> retrieve -> rerank -> Z_t.

Runs independently of the numerical branch so a failure here can only remove the text edge.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from ..types import Card, NumericalState, PointInTimeData, TextEvidence
from .dedup import deduplicate
from .events import extract_events
from .llm import HouseClient
from .query import build_query
from .rerank import rerank_and_compress
from .retrieval import retrieve
from .semantic_signal import lexicon_signal, llm_signal


def text_pipeline(pit: PointInTimeData, state: NumericalState, card: Card, cfg: dict[str, Any],
                  deadline: float, client: HouseClient | None = None) -> TextEvidence:
    assets = state.grid.assets
    tcfg = cfg["text"]
    if not tcfg.get("enabled", True) or not pit.docs:
        return TextEvidence.empty(assets, "no documents available")
    try:
        ctx = [a for a in pit.histories if a not in assets][:12]
        ev = extract_events(pit.docs, assets, ctx, int(tcfg["chunk_chars"]))
        ev = deduplicate(ev, float(tcfg["dedup_jaccard"]), int(tcfg["dedup_days"]))
        terms, _ = build_query(state, card)
        ranked = retrieve(ev, terms, pit.asof, assets, tcfg["retrieval_weights"],
                          float(tcfg["recency_halflife_days"]))
        top = rerank_and_compress(ranked, int(tcfg["top_k"]), int(tcfg["token_budget_chars"]))
        if not top:
            return TextEvidence.empty(assets, "no relevant passages")
        mx = max(e.score for e in ranked) if ranked else 0.0
        quality = float(min(1.0, (sum(e.score for e in top[:3]) / max(3 * mx, 1e-9)))) if mx > 0 else 0.0
        client = client or HouseClient(cfg)
        sig = None
        notes = []
        if client.available and time.monotonic() < deadline - 20:
            sig = llm_signal(client, card, state, top, deadline, cfg)
            notes.append("llm ok" if sig is not None else "llm failed -> lexicon")
        else:
            notes.append("llm unavailable -> lexicon")
        if sig is None:
            sig = lexicon_signal(state, top)
        return TextEvidence(events=ev, retrieved=top, signal=sig, retrieval_quality=quality,
                            corroboration_max=max((e.corroboration for e in top), default=0),
                            llm_calls=client.calls, llm_ok=sig.source == "llm", notes=notes)
    except Exception as exc:  # text can only ever remove its own edge
        print(f"[agenthon] text pipeline degraded: {type(exc).__name__}: {exc}", file=sys.stderr)
        return TextEvidence.empty(assets, f"text error {type(exc).__name__}")


__all__ = ["text_pipeline"]
