"""L16 — numeric/text contradiction kappa in [0, 1].

kappa_a = confidence * |2 p_up - 1| * 1{sign(text) != sign(numeric trend)} * min(1, |trend z| / 1.5)
A confident text view against a clear numeric trend counts most; a weak trend or an unsure text
contradicts little. kappa widens the distribution (distribution.py) and lowers the gate.
"""

from __future__ import annotations

import numpy as np

from ..types import NumericalState, SemanticSignal


def contradiction_score(state: NumericalState, sig: SemanticSignal) -> tuple[float, dict[str, float]]:
    per: dict[str, float] = {}
    for a, s in sig.assets.items():
        st = state.assets.get(a)
        if st is None:
            continue
        tv = 2 * s.p_up - 1
        if abs(tv) < 1e-6 or abs(st.momentum_z) < 1e-6:
            per[a] = 0.0
            continue
        disagree = float(np.sign(tv) != np.sign(st.momentum_z))
        per[a] = float(sig.confidence * abs(tv) * disagree * min(1.0, abs(st.momentum_z) / 1.5))
    k = float(np.mean(list(per.values()))) if per else 0.0
    return float(np.clip(k, 0.0, 1.0)), per
