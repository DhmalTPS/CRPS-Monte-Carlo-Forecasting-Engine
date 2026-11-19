"""L8 — documents -> passage-level events with lexicon features.

A document can hold several events; we split into paragraph blocks of ~chunk_chars and keep the
ones with any market vocabulary. Each event carries hawk/dove tone, stress intensity, explicit
tail vocabulary, topics and the assets it plausibly touches.
"""

from __future__ import annotations

import re

from ..types import Event, TextDoc
from .lexicon import ASSET_TERMS, DOVISH, HAWKISH, STRESS, TAIL, TOPICS, asset_key

_WS = re.compile(r"[ \t]+")


def _count(text: str, terms) -> int:
    return sum(text.count(t) for t in terms)


def _passages(body: str, chunk: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out, cur = [], ""
    for p in paras:
        p = _WS.sub(" ", p.replace("\n", " "))
        if len(cur) + len(p) + 1 <= chunk:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                out.append(cur)
            while len(p) > chunk * 1.5:
                out.append(p[:chunk])
                p = p[chunk:]
            cur = p
    if cur:
        out.append(cur)
    return out


def extract_events(docs: list[TextDoc], card_assets: list[str], context_assets: list[str],
                   chunk_chars: int = 900, max_events: int = 3000) -> list[Event]:
    events: list[Event] = []
    keys = sorted({asset_key(a) for a in list(card_assets) + list(context_assets)})
    for d in docs:
        for k, psg in enumerate(_passages(d.text, chunk_chars)):
            low = psg.lower()
            hk, dv = _count(low, HAWKISH), _count(low, DOVISH)
            sr, tl = _count(low, STRESS), _count(low, TAIL)
            touched = tuple(a for a in keys if _count(low, ASSET_TERMS.get(a, ())) > 0)
            topics = tuple(t for t, terms in TOPICS.items() if _count(low, terms) > 0)
            if not touched and not topics and hk + dv + sr + tl == 0:
                continue
            tone = (hk - dv) / (hk + dv + 2.0)
            events.append(Event(
                event_id=f"{d.doc_id}#{k}", doc_id=d.doc_id, timestamp=d.timestamp,
                doc_type=d.doc_type, institution=d.institution, text=psg, topics=topics,
                assets=touched, hawk=float(tone), stress=float(min(1.0, sr / 6.0)), tail_terms=tl,
            ))
            if len(events) >= max_events:
                return events
    return events
