"""The open house model WILL emit malformed JSON; coercion may only weaken the signal."""

import pytest

from agenthon.errors import LLMParseError
from agenthon.text.schema import coerce_signal, extract_json


@pytest.mark.parametrize("raw", ["", "no json here", "{unterminated", "[1, 2, 3]"])
def test_unparseable(raw):
    with pytest.raises(LLMParseError):
        extract_json(raw)


def test_fenced_trailing_comma_and_think():
    obj = extract_json('<think>{"x":1}</think>```json\n{"confidence": "0.4", "assets": {},}\n```')
    assert obj["confidence"] == "0.4"


def test_string_with_braces_and_escapes():
    obj = extract_json('{"summary": "a \\"quoted\\" } brace", "confidence": 0.2}')
    assert obj["confidence"] == 0.2


def test_garbage_values_become_neutral():
    sig = coerce_signal({"assets": {"UST_2Y": {"p_up": "banana", "magnitude": "huge", "vol_view": 7}},
                         "confidence": 250, "scenarios": "none", "established": "yes"},
                        ["UST_2Y", "EUR"], set())
    a = sig.assets["UST_2Y"]
    assert a.p_up == 0.5 and a.magnitude == "none" and a.vol_view == "same"
    assert sig.assets["EUR"].p_up == 0.5 and sig.scenarios == [] and sig.established is False
    assert 0.0 <= sig.confidence <= 1.0
