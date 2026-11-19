"""The gate: g in [0,1], g = 0 for neutral text, monotone in evidence, capped for lexicon."""

import numpy as np

from agenthon.config import load_config
from agenthon.families import get_family_policy
from agenthon.fusion import information_gate
from agenthon.types import Event, SemanticSignal, TextEvidence


def _ev(conf, source="llm", q=0.8, corr=2):
    sig = SemanticSignal.neutral(["UST_2Y"], source)
    sig.confidence = conf
    e = Event("e", "d", "2020-01-01", "fomc_statement", "", "x", (), (), 0, 0, 0)
    return TextEvidence([e], [e], sig, q, corr)


def test_neutral_gives_exact_zero():
    cfg = load_config()
    pol = get_family_policy("T2-F1", cfg)
    assert information_gate(TextEvidence.empty(["UST_2Y"]), 0.0, pol, cfg).g == 0.0
    assert information_gate(_ev(0.9), 0.0, pol, cfg, action_strength=0.0).g == 0.0


def test_monotone_and_bounded():
    cfg = load_config()
    pol = get_family_policy("T2-F2", cfg)
    gs = [information_gate(_ev(c), 0.0, pol, cfg).g for c in np.linspace(0, 1, 6)]
    assert all(b >= a for a, b in zip(gs, gs[1:])) and all(0 <= g <= 1 for g in gs)
    hi = information_gate(_ev(0.9), 0.0, pol, cfg).g
    lo = information_gate(_ev(0.9), 0.9, pol, cfg).g
    assert lo < hi  # contradiction can only lower the gate


def test_lexicon_capped():
    cfg = load_config()
    pol = get_family_policy("T2-F2", cfg)
    assert information_gate(_ev(1.0, "lexicon"), 0.0, pol, cfg).g <= cfg["gate"]["lexicon_only_cap"] + 1e-12
