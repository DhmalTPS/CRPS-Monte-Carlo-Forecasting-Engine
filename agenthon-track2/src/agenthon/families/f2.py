"""F2 text-cued regime shift (incl. EM FX transfer): text is pivotal, tails secondary.

Section 9.1 mitigations: vol-standardised features (state.py), robust reweighting toward the
distribution-agnostic walk (transfer model), and a higher gate prior on text.
"""

from .base import FamilyPolicy


def policy(cfg: dict, n_draws: int) -> FamilyPolicy:
    f = cfg.get("families", {}).get("f2", {})
    return FamilyPolicy(
        family="T2-F2", n_draws=n_draws,
        gate_intercept_shift=float(f.get("gate_intercept_shift", 0.4)),
        text_strength=float(f.get("text_strength", 1.0)),
        tail_strength=float(f.get("tail_strength", 1.0)),
        dependence_strength=float(f.get("dependence_strength", 0.5)),
        scale_mult=float(f.get("scale_mult", 1.0)),
        notes="regime shift lives in the text before the panel",
    )
