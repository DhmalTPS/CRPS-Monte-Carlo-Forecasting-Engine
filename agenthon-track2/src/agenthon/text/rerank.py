"""L12 — rerank with diversity (MMR) and compress into a fixed character budget.

Redundancy is penalised HERE (the MMR term), against what is already selected, which is the
right place: redundancy is a property of a set, not of a single passage.
"""

from __future__ import annotations

import re

from ..types import Event

_TOK = re.compile(r"[a-z0-9]+")


def _jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _compress(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sents:
        if len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
    return out or text[:max_chars]


def rerank_and_compress(ranked: list[Event], top_k: int = 12, budget_chars: int = 14000,
                        lam: float = 0.7, pool: int = 60) -> list[Event]:
    cand = ranked[:pool]
    toks = [set(_TOK.findall(e.text.lower())) for e in cand]
    chosen: list[int] = []
    per = max(400, budget_chars // max(top_k, 1))
    used = 0
    while len(chosen) < min(top_k, len(cand)):
        best, bi = -1e9, -1
        for i, e in enumerate(cand):
            if i in chosen:
                continue
            red = max((_jac(toks[i], toks[j]) for j in chosen), default=0.0)
            v = lam * e.score - (1 - lam) * 3.0 * red
            if v > best:
                best, bi = v, i
        if bi < 0:
            break
        e = cand[bi]
        e.text = _compress(e.text, per)
        if used + len(e.text) > budget_chars:
            break
        used += len(e.text)
        chosen.append(bi)
    out = [cand[i] for i in chosen]
    return sorted(out, key=lambda e: e.timestamp)
