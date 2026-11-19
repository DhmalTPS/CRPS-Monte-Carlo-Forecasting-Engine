"""F3 cross-asset reasoning: the joint variogram term is load-bearing; dependence gets the effort."""

from .base import FamilyPolicy


def policy(cfg: dict, n_draws: int) -> FamilyPolicy:
    f = cfg.get("families", {}).get("f3", {})
    return FamilyPolicy(
        family="T2-F3", n_draws=n_draws,
        gate_intercept_shift=float(f.get("gate_intercept_shift", 0.0)),
        text_strength=float(f.get("text_strength", 0.8)),
        tail_strength=float(f.get("tail_strength", 0.6)),
        dependence_strength=float(f.get("dependence_strength", 1.0)),
        scale_mult=float(f.get("scale_mult", 1.0)),
        notes="apply text consistently across all assets within each draw",
    )
