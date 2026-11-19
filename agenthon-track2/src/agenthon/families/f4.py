"""F4 tail/shock-from-text: the tail term is load-bearing; >= 1000 draws; event-triggered tails."""

from .base import FamilyPolicy


def policy(cfg: dict, n_draws: int) -> FamilyPolicy:
    f = cfg.get("families", {}).get("f4", {})
    df = f.get("df")
    return FamilyPolicy(
        family="T2-F4", n_draws=max(n_draws, int(cfg["sampling"].get("n_f4", 4000))),
        gate_intercept_shift=float(f.get("gate_intercept_shift", 0.2)),
        text_strength=float(f.get("text_strength", 0.9)),
        tail_strength=float(f.get("tail_strength", 1.3)),
        dependence_strength=float(f.get("dependence_strength", 0.8)),
        scale_mult=float(f.get("scale_mult", 1.0)),
        df=float(df) if df is not None else None,
        notes="missing a shock vs over-inflating ordinary-day tails",
    )
