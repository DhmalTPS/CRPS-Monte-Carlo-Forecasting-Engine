"""L15 — information gate g_t in [0, 1] (eq. 18), monotone by construction.

g = sigmoid(w0 + dw0_family + w1*retrieval + w2*confidence + w3*log(1 + corroboration) - w4*kappa)
with w1..w4 >= 0 enforced (abs), so better evidence can only raise g and contradiction can only
lower it. Hard rules: a neutral signal gives g = 0 EXACTLY (=> P1 == P0, eq. 11); a lexicon-only
signal is capped; the family's text_strength and the policy's action multiply g.
"""

from __future__ import annotations

import math
from typing import Any

from ..families.base import FamilyPolicy
from ..types import GateDecision, TextEvidence


def information_gate(ev: TextEvidence, kappa: float, policy: FamilyPolicy, cfg: dict[str, Any],
                     action_strength: float = 1.0) -> GateDecision:
    gc, uc = cfg["gate"], cfg["update"]
    sig = ev.signal
    if sig.source == "neutral" or not ev.retrieved or action_strength <= 0:
        return GateDecision(0.0, kappa, "none", {"reason": 0.0})
    x = (float(gc["w0"]) + policy.gate_intercept_shift
         + abs(float(gc["w_retrieval"])) * ev.retrieval_quality
         + abs(float(gc["w_confidence"])) * sig.confidence
         + abs(float(gc["w_corroboration"])) * math.log1p(max(ev.corroboration_max, 0))
         - abs(float(gc["w_kappa"])) * kappa)
    g = 1.0 / (1.0 + math.exp(-x))
    if sig.source == "lexicon":
        g = min(g, float(gc.get("lexicon_only_cap", 0.25)))
    g *= float(policy.text_strength) * float(action_strength)
    g = float(min(max(g, 0.0), float(gc.get("g_max", 0.85))))
    mech = "mixture" if (policy.allow_mixture and (len(sig.scenarios) >= 2 or
                                                   kappa > float(uc.get("kappa_star", 0.55)))) else "tilt"
    return GateDecision(g, kappa, mech if g > 0 else "none", {"logit": x})
