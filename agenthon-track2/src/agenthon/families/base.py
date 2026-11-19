"""Family policy: one shared backbone, a handful of parameters that differ by family.

Never four pipelines. A policy only scales existing knobs (docs Section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FamilyPolicy:
    family: str
    n_draws: int = 2000
    gate_intercept_shift: float = 0.0  # added to w0 of the information gate (F2: text is pivotal)
    text_strength: float = 1.0  # multiplier on text-driven shifts
    tail_strength: float = 1.0  # multiplier on event-triggered tail inflation (F4)
    dependence_strength: float = 1.0  # multiplier on text-driven correlation shift (F3)
    scale_mult: float = 1.0  # multiplicative width prior for the family (validated offline)
    df: float | None = None  # override the scale-mixture dof (None -> config)
    allow_mixture: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
