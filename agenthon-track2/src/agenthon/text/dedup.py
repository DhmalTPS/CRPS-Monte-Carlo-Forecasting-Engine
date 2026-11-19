"""L9 — event deduplication: k wire copies of one headline are ONE piece of evidence.

Two passages merge iff their word-5-gram Jaccard >= theta AND their timestamps are within
dedup_days (semantic AND temporal proximity, eq. 14). Union-find clusters; the cluster's
corroboration count is the number of distinct documents merged; novelty nu = 1 / (1 + log c)
[uncertain — empirical validation required].
"""

from __future__ import annotations

import datetime as dt
import math
import re

from ..types import Event

_TOK = re.compile(r"[a-z0-9]+")


def _shingles(text: str, n: int = 5) -> set[int]:
    toks = _TOK.findall(text.lower())
    if len(toks) < n:
        return {hash(" ".join(toks))}
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def _days(a: str, b: str) -> float:
    try:
        return abs((dt.date.fromisoformat(a[:10]) - dt.date.fromisoformat(b[:10])).days)
    except ValueError:
        return 1e9


def deduplicate(events: list[Event], theta: float = 0.55, max_days: int = 5) -> list[Event]:
    n = len(events)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sh = [_shingles(e.text) for e in events]
    order = sorted(range(n), key=lambda i: events[i].timestamp)
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if _days(events[i].timestamp, events[j].timestamp) > max_days:
                break
            if events[i].doc_id == events[j].doc_id:
                continue
            a, b = sh[i], sh[j]
            inter = len(a & b)
            if inter and inter / len(a | b) >= theta:
                parent[find(j)] = find(i)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    out: list[Event] = []
    for cid, members in enumerate(clusters.values()):
        docs = {events[m].doc_id for m in members}
        rep = max(members, key=lambda m: (len(events[m].text), events[m].timestamp))
        e = events[rep]
        e.cluster = cid
        e.corroboration = len(docs)
        e.novelty = 1.0 / (1.0 + math.log(max(len(docs), 1)))
        out.append(e)
    return out
