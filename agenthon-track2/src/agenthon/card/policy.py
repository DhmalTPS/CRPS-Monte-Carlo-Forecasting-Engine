"""L0 -> family policy. One backbone; families differ only through a handful of parameters."""

from __future__ import annotations

from typing import Any

from ..families import get_family_policy
from ..families.base import FamilyPolicy
from ..types import Card


def resolve_policy(card: Card, cfg: dict[str, Any]) -> FamilyPolicy:
    return get_family_policy(card.family, cfg, card)
