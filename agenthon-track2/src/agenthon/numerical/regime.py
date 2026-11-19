"""L4 — a deliberately small regime model (K = 3: calm / normal / stressed).

With ~100 cards and thin per-regime samples a high-K latent model cannot be estimated without
heavy shrinkage, so the regime is a *soft* function of one well-measured quantity: the ratio of
current (EWMA) to long-run volatility, averaged over the card's assets on a log scale. Soft
membership (logistic in log-ratio) avoids hard-threshold discontinuities. On failure: one global
regime, which reduces every regime-conditional formula to its unconditional form.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import AssetState, RegimeBelief

LABELS = ("calm", "normal", "stressed")


def estimate_regime(states: dict[str, AssetState], cfg: dict[str, Any]) -> RegimeBelief:
    try:
        r = np.array([max(s.vol_ratio, 1e-6) for s in states.values()])
        lr = float(np.mean(np.log(r)))
        lo = np.log(float(cfg["regime"].get("calm_ratio", 0.8)))
        hi = np.log(float(cfg["regime"].get("stress_ratio", 1.35)))
        k = 8.0
        p_stress = 1 / (1 + np.exp(-k * (lr - hi)))
        p_calm = 1 / (1 + np.exp(k * (lr - lo)))
        p_norm = max(1.0 - p_stress - p_calm, 0.0)
        p = np.array([p_calm, p_norm, p_stress], dtype=float)
        p = p / p.sum()
        return RegimeBelief(LABELS, p, "numeric")
    except Exception:
        return RegimeBelief(("normal",), np.array([1.0]), "global")


def update_regime_with_text(regime: RegimeBelief, evidence: str, g: float) -> RegimeBelief:
    """L17 — pi' ∝ pi ⊙ exp(g * rho), rho encodes which regime the text supports."""
    if len(regime.probs) != 3 or g <= 0:
        return regime
    rho = {"supports_stress": np.array([-1.0, 0.0, 1.0]),
           "supports_calm": np.array([1.0, 0.0, -1.0])}.get(evidence)
    if rho is None:
        return regime
    p = regime.probs * np.exp(g * rho)
    return RegimeBelief(regime.labels, p / p.sum(), "numeric+text")
