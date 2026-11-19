"""F1 continuation-with-context: marginal CRPS dominates; text is incremental, gate mostly small."""

from .base import FamilyPolicy


def policy(cfg: dict, n_draws: int) -> FamilyPolicy:
    f = cfg.get("families", {}).get("f1", {})
    return FamilyPolicy(
        family="T2-F1", n_draws=n_draws,
        gate_intercept_shift=float(f.get("gate_intercept_shift", -0.3)),
        text_strength=float(f.get("text_strength", 0.8)),
        tail_strength=float(f.get("tail_strength", 0.5)),
        dependence_strength=float(f.get("dependence_strength", 0.5)),
        scale_mult=float(f.get("scale_mult", 1.0)),
        notes="numbers carry most of the signal; avoid over-engineering text",
    )
