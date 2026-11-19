"""Text branch end to end with a FAKE House endpoint (no network): the LLM answer flows to Z."""

import json

from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.numerical import build_grid, build_state
from agenthon.safety import Deadline
from agenthon.text import text_pipeline


class FakeClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0
        self.available = True

    def chat(self, messages, max_tokens=None, deadline=None):
        self.calls += 1
        assert "AS-OF DATE" in messages[1]["content"]
        return self.answer


def _setup(unit):
    c = parse_card(unit / "card.toml")
    cfg = load_config()
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, cfg)
    g = build_grid(c, pit.histories)
    return c, cfg, pit, build_state(c, g, pit.histories, cfg)


def test_llm_signal_parsed(unit):
    c, cfg, pit, st = _setup(unit)
    ans = json.dumps({"assets": {"UST_2Y": {"p_up": 0.7, "magnitude": "moderate", "vol_view": "higher",
                                            "tail_side": "up"},
                                 "UST_10Y": {"p_up": 0.6, "magnitude": "weak", "vol_view": "same",
                                             "tail_side": "none"}},
                      "scenarios": [], "persistence": "medium", "regime_evidence": "supports_stress",
                      "confidence": 0.6, "novelty": 0.5, "established": False,
                      "cited_docs": ["fomc_statement_20200610"], "summary": "hawkish"})
    ev = text_pipeline(pit, st, c, cfg, Deadline(300).end, FakeClient("<think>x</think>" + ans))
    assert ev.llm_ok and ev.signal.source == "llm"
    assert ev.signal.assets["UST_2Y"].p_up == 0.7 and ev.signal.cited_docs == ("fomc_statement_20200610",)


def test_bad_json_falls_back_to_lexicon(unit):
    c, cfg, pit, st = _setup(unit)
    ev = text_pipeline(pit, st, c, cfg, Deadline(300).end, FakeClient("I think rates go up."))
    assert ev.signal.source == "lexicon" and not ev.llm_ok
