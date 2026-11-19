"""House endpoint down / refusing / timing out: controlled loss of the text edge, never a crash."""

import pytest

from agenthon.errors import LLMUnavailable
from agenthon.main import run
from agenthon.text.llm import HouseClient


class Down:
    calls = 0
    available = True

    def chat(self, *a, **k):
        self.calls += 1
        raise LLMUnavailable("HTTP 503")


def test_endpoint_unreachable(monkeypatch, unit, out_dir):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://127.0.0.1:9")  # discard port: refused
    monkeypatch.setenv("MODEL_TOKEN", "t")
    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                    overrides={"llm": {"timeout_seconds": 5, "max_calls": 1}})
    assert path == "full" and res.diagnostics["text"]["source"] in ("lexicon", "neutral")


def test_client_budget_and_offline(monkeypatch):
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    c = HouseClient({"llm": {"enabled": True, "max_calls": 1}})
    assert not c.available
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "x"}])


def test_down_client_injected(unit, out_dir):
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet", client=Down())
    assert path == "full" and not res.diagnostics["text"]["source"] == "llm"
