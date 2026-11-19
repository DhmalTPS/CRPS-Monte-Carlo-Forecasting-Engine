"""L0 — semantic validation of the parsed card and resolution of the information cutoff."""

from __future__ import annotations

import re
import sys

from ..errors import CardError
from ..types import Card

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_card(card: Card, cli_asof: str | None) -> str:
    """Return the information cutoff to use (ISO date). Raise CardError only if unusable.

    forecast_meta.json must echo the card's trusted as-of (g2 binds it), so `card.asof` is what we
    *write*; the information set is the earlier of the two dates, which can only ever narrow it.
    """
    if not card.unit_id:
        raise CardError("empty [task].id")
    if not card.assets or not card.horizons:
        raise CardError("card declares no assets or horizons")
    if any(h <= 0 for h in card.horizons):
        raise CardError("non-positive horizon")
    if len(set(card.assets)) != len(card.assets) or len(set(card.horizons)) != len(card.horizons):
        raise CardError("duplicate asset or horizon in the grid")
    if card.target_type not in ("level", "log_return"):
        print(f"[agenthon] unknown target_type {card.target_type!r}; treating as level",
              file=sys.stderr)
    asof = card.asof if _ISO.match(card.asof or "") else ""
    cli = cli_asof if (cli_asof and _ISO.match(cli_asof)) else ""
    if not asof and not cli:
        raise CardError("no usable as-of date on the card or the command line")
    if asof and cli and cli != asof:
        print(f"[agenthon] --asof {cli} differs from card data_cutoff {asof}; using the earlier "
              "as the information cutoff and the card's value in forecast_meta.json",
              file=sys.stderr)
    return min(d for d in (asof, cli) if d)
