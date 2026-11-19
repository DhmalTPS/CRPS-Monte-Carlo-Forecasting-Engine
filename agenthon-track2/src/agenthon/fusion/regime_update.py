"""L17 — text-adjusted regime posterior (Conditional; identity when regime conditioning is off)."""

from __future__ import annotations

from ..numerical.regime import update_regime_with_text
from ..types import GateDecision, RegimeBelief, SemanticSignal


def regime_update(regime: RegimeBelief, sig: SemanticSignal, gate: GateDecision) -> RegimeBelief:
    return update_regime_with_text(regime, sig.regime_evidence, gate.g)
