"""L11 — single scored retrieval (the five reference-draft channels MERGED into eq. 16):

score(e | q) = a*BM25 + sqrt(BM25) * [b*exp(-lambda*(t - tau)) + c*asset_match + d*novelty
               + f*doc_type_weight]   (redundancy -e applied set-wise by MMR)

BM25 index is built at run time over THIS unit's passages only (no cross-unit memory).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter

from ..types import Event
from .lexicon import DOC_TYPE_WEIGHT, asset_key

_TOK = re.compile(r"[a-z0-9&']+")


def _tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def bm25_scores(events: list[Event], query_terms: list[str], k1: float = 1.4,
                b: float = 0.75) -> list[float]:
    docs = [_tok(e.text) for e in events]
    if not docs:
        return []
    N = len(docs)
    avg = sum(len(d) for d in docs) / N
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q_uni = [t for t in query_terms if " " not in t]
    q_phr = [t for t in query_terms if " " in t]
    out = []
    for e, d in zip(events, docs):
        tf = Counter(d)
        s = 0.0
        for t in q_uni:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        low = e.text.lower()
        s += sum(1.5 for p in q_phr if p in low)
        out.append(s)
    return out


def retrieve(events: list[Event], query_terms: list[str], asof: str, assets: list[str],
             weights: dict[str, float], halflife_days: float = 45.0) -> list[Event]:
    if not events:
        return []
    raw = bm25_scores(events, query_terms)
    mx = max(raw) or 1.0
    keys = {asset_key(a) for a in assets}
    t0 = dt.date.fromisoformat(asof[:10])
    lam = math.log(2) / max(halflife_days, 1.0)
    for e, r in zip(events, raw):
        try:
            age = max((t0 - dt.date.fromisoformat(e.timestamp[:10])).days, 0)
        except ValueError:
            age = 3650
        match = len(keys & set(e.assets)) / max(len(keys), 1)
        rel = r / mx
        # relevance gates the secondary terms: recency, asset match, novelty and source type can
        # re-order relevant passages but cannot lift an irrelevant one (score 0 at zero BM25)
        e.score = (weights.get("bm25", 1.0) * rel
                   + math.sqrt(rel) * (weights.get("recency", 0.6) * math.exp(-lam * age)
                                       + weights.get("asset", 0.8) * match
                                       + weights.get("novelty", 0.3) * e.novelty
                                       + weights.get("doc_type", 0.4) * DOC_TYPE_WEIGHT.get(e.doc_type, 0.5)))
        # (the redundancy term of eq. 16 is applied set-wise by MMR in rerank.py)
    return sorted(events, key=lambda e: -e.score)
