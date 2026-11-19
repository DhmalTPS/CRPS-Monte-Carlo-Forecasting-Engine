"""The House HTTP contract against a local mock server (no internet): path, auth, body, fallback."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agenthon.main import run
from agenthon.text.llm import HouseClient

SEEN: list[dict] = []
ANSWER = {"assets": {"UST_2Y": {"p_up": 0.62, "magnitude": "weak", "vol_view": "higher", "tail_side": "up"},
                     "UST_10Y": {"p_up": 0.55, "magnitude": "weak", "vol_view": "same", "tail_side": "none"}},
          "scenarios": [], "persistence": "medium", "regime_evidence": "supports_stress",
          "confidence": 0.55, "novelty": 0.4, "established": False,
          "cited_docs": ["fomc_statement_20200610"], "summary": "tightening bias, market stress"}


class Handler(BaseHTTPRequestHandler):
    status = 200

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.path != "/v1/chat/completions":
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(Handler.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if Handler.status == 200:
            self.wfile.write(json.dumps({"choices": [{"message": {"content": json.dumps(ANSWER)}}]}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def house(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("MODEL_ENDPOINT", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("MODEL_TOKEN", "unit-token")
    monkeypatch.setenv("MODEL_NAME", "house")
    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(v, raising=False)
    SEEN.clear()
    Handler.status = 200
    yield srv
    srv.shutdown()


def test_contract_and_signal_flow(house, unit, out_dir):
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet")
    assert path == "full" and res.diagnostics["text"]["source"] == "llm"
    assert len(SEEN) == 1
    call = SEEN[0]
    assert call["path"] == "/v1/chat/completions" and call["auth"] == "Bearer unit-token"
    b = call["body"]
    assert b["model"] == "house" and b["n"] == 1 and b["max_tokens"] <= 4000
    assert b["temperature"] == 0.0 and b["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in b  # vendor-side tools are never requested
    assert res.diagnostics["text"]["gate"] > 0
    assert "fomc_statement_20200610" in (out_dir / "forecast_rationale.md").read_text(encoding="utf-8")


def test_upstream_failure_degrades(house, unit, out_dir):
    Handler.status = 503
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet")
    assert path == "full" and res.diagnostics["text"]["source"] == "lexicon"
    assert len(SEEN) == 1  # a failed admitted request is not blindly retried (it costs a slot)


def test_call_cap(house):
    c = HouseClient({"llm": {"enabled": True, "max_calls": 2, "timeout_seconds": 10, "max_tokens": 100}})
    for _ in range(2):
        c.chat([{"role": "user", "content": "x"}])
    with pytest.raises(Exception):
        c.chat([{"role": "user", "content": "x"}])
    assert len(SEEN) == 2


class Strict(Handler):
    """A deployment that rejects optional OpenAI fields with HTTP 400."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "body": body})
        if "seed" in body or "chat_template_kwargs" in body:
            self.send_response(400)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": json.dumps(ANSWER)}}]}).encode())


def test_400_retries_with_minimal_body(monkeypatch, unit, out_dir):
    srv = HTTPServer(("127.0.0.1", 0), Strict)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MODEL_ENDPOINT", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(v, raising=False)
    SEEN.clear()
    try:
        path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet")
    finally:
        srv.shutdown()
    assert path == "full" and res.diagnostics["text"]["source"] == "llm"
    assert len(SEEN) == 2 and "seed" not in SEEN[1]["body"] and SEEN[1]["body"]["max_tokens"] == 4000
