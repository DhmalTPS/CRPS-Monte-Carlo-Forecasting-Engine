"""Top-level fail-safe: whatever happened, write an admissible forecast and exit 0.

A failed card is scored 4.0; M0's own draws are scored exactly 1.0. This is the single most
valuable piece of engineering in the system (Section 8.20), so it depends on nothing but the card
and the panels, and has its own fallbacks.
"""

from __future__ import annotations

import pathlib
import sys
import traceback

from ..output import build_meta, cleanup_temporaries, fallback_rationale, write_forecast
from ..types import Card


def fail_safe_forecast(card: Card | None, panels_dir: pathlib.Path, out_path: pathlib.Path,
                       exc: BaseException | None) -> str:
    reason = f"{type(exc).__name__}: {exc}" if exc is not None else "unknown"
    print(f"[agenthon] FAIL-SAFE engaged: {reason}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    cleanup_temporaries(out_path.parent)
    if card is None:
        raise SystemExit(f"card unreadable; cannot emit a bound forecast ({reason})")
    from .fallback import constant_fallback, m0_fallback

    try:
        draws, grid, path = m0_fallback(card, panels_dir)
    except Exception as exc2:
        print(f"[agenthon] panel-based fallbacks failed ({type(exc2).__name__}); constant", file=sys.stderr)
        import re

        anchors = {}
        for a in card.assets:
            m = re.search(rf"\b{re.escape(a)}\s*=\s*(-?\d+(?:\.\d+)?)", card.prose + " " + card.description)
            if m:
                anchors[a] = float(m.group(1))
        draws, grid = constant_fallback(card, anchors)
        path = "failsafe_constant"
    meta = build_meta(card, draws.shape[0], f"fail-safe {path}", 0)
    write_forecast(out_path, draws, grid, card, meta,
                   fallback_rationale(card.unit_id, card.asof, reason[:300], path))
    return path
