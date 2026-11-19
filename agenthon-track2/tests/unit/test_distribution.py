"""L18: exact reduction at g=0, positivity of scale, validity of the mixture, no inferred narrowing."""

import numpy as np

from agenthon.config import load_config
from agenthon.families import get_family_policy
from agenthon.fusion import distribution_update
from agenthon.sampling import sample_belief
from agenthon.types import (AssetSignal, Cell, GateDecision, Grid, JointBelief, Scenario,
                            ScenarioView, SemanticSignal)

ABSZ = {"weak": 0.674, "moderate": 1.15, "strong": 1.645, "extreme": 2.33}


def _p0():
    g = Grid((Cell("A", 21, 21), Cell("A", 63, 63)))
    return JointBelief(g, np.array([1.0, 1.0]), (21, 63), {"A": np.array([1.0, 3.0])}, np.eye(1), 1000.0,
                       [Scenario("base", 1.0, np.zeros(2), np.ones(2))])


def _sig(p_up=0.8, vol="higher", mag="strong", scen=()):
    s = SemanticSignal.neutral(["A"], "llm")
    s.assets["A"] = AssetSignal("A", p_up, 1 - p_up, mag, vol, "none")
    s.scenarios = list(scen)
    s.confidence = 0.8
    return s


def test_zero_gate_is_identity():
    cfg = load_config()
    p0 = _p0()
    p1, _ = distribution_update(p0, _sig(), GateDecision(0.0, 0.0, "none"), ABSZ,
                                get_family_policy("T2-F1", cfg), cfg)
    assert p1 is p0


def test_tilt_moves_centre_up_and_widens():
    cfg = load_config()
    p1, _ = distribution_update(_p0(), _sig(), GateDecision(0.5, 0.0, "tilt"), ABSZ,
                                get_family_policy("T2-F2", cfg), cfg)
    assert np.all(p1.mu > 1.0) and np.all(p1.scenarios[0].scale > 1.0)


def test_inferred_lower_vol_does_not_narrow():
    cfg = load_config()
    p1, _ = distribution_update(_p0(), _sig(vol="lower"), GateDecision(0.6, 0.0, "tilt"), ABSZ,
                                get_family_policy("T2-F2", cfg), cfg)
    assert np.all(p1.scenarios[0].scale >= 1.0)


def test_mixture_weights_valid_and_sampleable():
    cfg = load_config()
    sv = [ScenarioView("hike", 0.6, {"A": 1}, "strong", "higher"), ScenarioView("hold", 0.4, {"A": 0}, "weak", "same")]
    p1, _ = distribution_update(_p0(), _sig(scen=sv), GateDecision(0.5, 0.7, "mixture"), ABSZ,
                                get_family_policy("T2-F2", cfg), cfg)
    w = np.array([s.weight for s in p1.scenarios])
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9
    d = sample_belief(p1, 2000, 0)
    assert np.all(np.isfinite(d)) and d.shape == (2000, 2)


def test_text_widening_is_capped():
    cfg = load_config()
    p1, _ = distribution_update(_p0(), _sig(vol="much_higher"), GateDecision(0.85, 1.0, "tilt"), ABSZ,
                                get_family_policy("T2-F2", cfg), cfg)
    assert np.all(p1.scenarios[0].scale <= cfg["update"]["max_text_vol_mult"] + 1e-12)
