from __future__ import annotations

from typing import Any

from . import f1, f2, f3, f4
from .base import FamilyPolicy

_MAP = {"T2-F1": f1.policy, "T2-F2": f2.policy, "T2-F3": f3.policy, "T2-F4": f4.policy}


def get_family_policy(family: str, cfg: dict[str, Any], card: Any = None) -> FamilyPolicy:
    n = int(cfg["sampling"].get("n_default", 2000))
    if card is not None:
        n = max(n, int(getattr(card, "n_draws_min", 200)))
    fn = _MAP.get(family)
    if fn is None:
        return FamilyPolicy(family=family or "T2-UNKNOWN", n_draws=n,
                            notes="unknown family: neutral policy")
    return fn(cfg, n)


__all__ = ["FamilyPolicy", "get_family_policy"]
