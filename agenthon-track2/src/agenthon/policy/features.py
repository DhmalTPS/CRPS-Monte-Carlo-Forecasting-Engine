"""Context vector for the adaptive policy. Small, vol-standardised, available at inference."""

from __future__ import annotations

import math

import numpy as np

from ..types import Card, NumericalState

CLASSES = ("rates", "fx", "factor", "macro", "em_fx")
FAMS = ("T2-F1", "T2-F2", "T2-F3", "T2-F4")
NAMES = ["bias", "log_vol_ratio", "abs_trend", "log_steps", "single_cell", "log_return"] + \
    [f"cls_{c}" for c in CLASSES] + [f"fam_{f}" for f in FAMS] + ["text_gate"]


def context_vector(card: Card, state: NumericalState, text_gate: float = 0.0) -> np.ndarray:
    sts = list(state.assets.values())
    lvr = float(np.mean([math.log(max(s.vol_ratio, 1e-6)) for s in sts]))
    tr = float(np.mean([min(abs(s.momentum_z), 4.0) for s in sts]))
    steps = float(np.mean([c.steps for c in state.grid.cells]))
    cls = [float(np.mean([s.asset_class == c for s in sts])) for c in CLASSES]
    fam = [float(card.family == f) for f in FAMS]
    return np.array([1.0, lvr, tr, math.log(max(steps, 1) / 21.0), float(card.is_single_cell),
                     float(card.target_type == "log_return")] + cls + fam + [float(text_gate)])
