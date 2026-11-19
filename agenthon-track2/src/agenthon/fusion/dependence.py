"""L19 (text part) — convex move of the increment correlation toward a stress structure.

Only when the text supports a stress regime and the card has more than one asset. PSD is
preserved by convexity (Proposition 3). g = 0 => identity.
"""

from __future__ import annotations

from ..sampling.dependence import blend_correlation, stress_correlation
from ..types import GateDecision, JointBelief, SemanticSignal


def dependence_update(belief: JointBelief, sig: SemanticSignal, gate: GateDecision,
                      strength: float = 1.0) -> tuple[JointBelief, float]:
    if gate.g <= 0 or belief.corr.shape[0] < 2 or sig.regime_evidence != "supports_stress":
        return belief, 0.0
    w = float(min(0.5, 0.5 * gate.g * strength))
    out = belief.copy()
    out.corr = blend_correlation(belief.corr, stress_correlation(belief.corr, 0.5), w)
    return out, w
