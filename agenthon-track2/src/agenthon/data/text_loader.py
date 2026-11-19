"""L1 — load the frozen, dated text corpus for THIS unit only (no cross-unit memory).

Timestamps come from text/corpus_index.json (release dates). A file absent from the index is
dated from its filename; a document whose availability cannot be established is dropped.
"""

from __future__ import annotations

import json
import pathlib
import sys

from ..types import TextDoc
from .availability import date_from_filename, doc_available

_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".json"}


def load_corpus(text_dir: pathlib.Path, asof: str, max_docs: int = 60,
                max_chars: int = 120000) -> tuple[list[TextDoc], int]:
    docs: list[TextDoc] = []
    dropped = 0
    try:
        if not text_dir.is_dir():
            return [], 0
    except OSError:
        return [], 0
    index: dict[str, dict] = {}
    idx_path = text_dir / "corpus_index.json"
    if idx_path.is_file():
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8", errors="replace"))
            for row in data.get("documents", []) if isinstance(data, dict) else []:
                if isinstance(row, dict) and row.get("file"):
                    index[str(row["file"])] = row
        except (ValueError, OSError):
            print("[agenthon] corpus_index.json unreadable; dating from filenames", file=sys.stderr)
    try:
        files = sorted(p for p in text_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES
                       and p.name != "corpus_index.json")
    except OSError:
        return [], 0
    for p in files:
        meta = index.get(p.name, {})
        ts = str(meta.get("timestamp") or "")[:10] or date_from_filename(p.name)
        if not doc_available(ts, asof):
            dropped += 1
            continue
        try:
            body = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError:
            dropped += 1
            continue
        if not body.strip():
            continue
        docs.append(
            TextDoc(
                doc_id=str(meta.get("doc_id") or p.stem),
                timestamp=str(ts),
                doc_type=str(meta.get("doc_type") or _guess_type(p.name)),
                source=str(meta.get("source") or ""),
                institution=str(meta.get("institution") or _guess_institution(p.name, body)),
                text=body,
                path=str(p),
            )
        )
    docs.sort(key=lambda d: (d.timestamp, d.doc_id))
    if len(docs) > max_docs:  # keep the most recent (most relevant to the forecast origin)
        dropped += len(docs) - max_docs
        docs = docs[-max_docs:]
    return docs, dropped


def _guess_type(name: str) -> str:
    n = name.lower()
    for key, t in (("minutes", "fomc_minutes"), ("statement", "fomc_statement"),
                   ("beige", "beige_book"), ("cpi", "macro_release"), ("empsit", "macro_release"),
                   ("nfp", "macro_release"), ("cot", "cot"), ("bis_", "cb_speech"),
                   ("speech", "cb_speech"), ("testimony", "testimony")):
        if key in n:
            return t
    return "document"


def _guess_institution(name: str, body: str) -> str:
    n = (name + " " + body[:400]).lower()
    for key, inst in (("fomc", "Federal Reserve"), ("federal reserve", "Federal Reserve"),
                      ("ecb", "European Central Bank"), ("european central bank", "European Central Bank"),
                      ("bank of japan", "Bank of Japan"), ("bank of england", "Bank of England"),
                      ("reserve bank of australia", "Reserve Bank of Australia"),
                      ("bank of canada", "Bank of Canada"), ("swiss national bank", "Swiss National Bank"),
                      ("people's bank of china", "People's Bank of China"),
                      ("bureau of labor statistics", "BLS")):
        if key in n:
            return inst
    return ""
