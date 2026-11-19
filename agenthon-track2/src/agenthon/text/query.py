"""L10 — state-conditioned query q_t = phi(s_t, pi_t, h, A).

Not "recent news": the query names the target assets' vocabulary, the numeric regime (a stressed
state pulls stress vocabulary in), the direction of the recent move (what the text should explain
or contradict), and the card's own authored description/notes (a legitimate /input file).
"""

from __future__ import annotations

import re

from ..types import Card, NumericalState
from .lexicon import ASSET_TERMS, STRESS, TAIL, asset_key

_STOP = set("the a an of and or to in on for with by at from is are be as that this it its their "
            "will would can could may might should into than then there these those over under "
            "whether which what when how card target targets horizon horizons forecast bd".split())


def build_query(state: NumericalState, card: Card) -> tuple[list[str], str]:
    terms: list[str] = []
    for a in state.grid.assets:
        terms += list(ASSET_TERMS.get(asset_key(a), ()))
    if state.regime.stressed > 0.4:
        terms += list(STRESS) + list(TAIL)
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{3,}", " ".join([card.title, card.description,
                                                              card.text_notes]).lower())
    terms += [w for w in words if w not in _STOP][:60]
    moves = []
    for a, st in state.assets.items():
        dirn = "rising" if st.momentum_z > 0.5 else "falling" if st.momentum_z < -0.5 else "flat"
        moves.append(f"{a} {dirn} (3m trend z={st.momentum_z:+.2f}, vol ratio={st.vol_ratio:.2f})")
    reg = ", ".join(f"{l}={p:.2f}" for l, p in zip(state.regime.labels, state.regime.probs))
    text = (f"Targets {', '.join(state.grid.assets)} at horizons "
            f"{sorted({c.horizon for c in state.grid.cells})} business days. Regime: {reg}. "
            f"Recent moves: {'; '.join(moves)}.")
    return sorted(set(t.lower() for t in terms)), text
