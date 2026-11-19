"""L13 — the House model client (OpenAI-compatible chat completions via the audited proxy).

Contract (SUBMISSION_CLI.md / HOUSE-MODEL.md):
  POST $MODEL_ENDPOINT/v1/chat/completions, Authorization: Bearer $MODEL_TOKEN, model=$MODEL_NAME.
  25 admitted requests per unit (charged at admission, retries cost a slot), <= 4000 output
  tokens per call, no n>1, vendor tools off. Proxies are read from the environment by urllib.
We cap ourselves far below the allowance (cfg llm.max_calls), pin temperature/seed, disable
thinking (thinking tokens would eat the 4000-token output cap), and treat every failure as
"text unavailable" -> neutral signal -> gate 0 -> P1 == P0.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from ..errors import BudgetExceeded, LLMUnavailable


class HouseClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["llm"]
        self.endpoint = (os.environ.get("MODEL_ENDPOINT") or "").rstrip("/")
        self.token = os.environ.get("MODEL_TOKEN") or ""
        self.model = os.environ.get("MODEL_NAME") or "house"
        self.calls = 0
        self.network = (os.environ.get("QFBENCH_NETWORK") or "").lower()
        self.minimal = False  # set after a 400/422: resend without optional fields

    @property
    def available(self) -> bool:
        return bool(self.cfg.get("enabled", True) and self.endpoint and self.token
                    and self.network != "none")

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None,
             deadline: float | None = None) -> str:
        if not self.available:
            raise LLMUnavailable("no House endpoint configured (offline/local run)")
        if self.calls >= int(self.cfg.get("max_calls", 3)):
            raise BudgetExceeded("own LLM call cap reached")
        timeout = float(self.cfg.get("timeout_seconds", 120))
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout < 10:
                raise BudgetExceeded("not enough wall-clock left for an LLM call")
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(min(max_tokens or self.cfg.get("max_tokens", 1800), 4000)),
            "temperature": float(self.cfg.get("temperature", 0.0)),
        }
        if self.minimal:
            body["max_tokens"] = 4000  # thinking may be on by default: leave room for the answer
        else:
            body.update({"seed": int(self.cfg.get("seed", 7)), "n": 1,
                         "chat_template_kwargs": {"enable_thinking": bool(self.cfg.get("enable_thinking", False))}})
        req = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        self.calls += 1  # charged at admission, even if the response is lost
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                self.calls -= 1  # refused before admission: not charged
            if exc.code in (400, 422) and not self.minimal:
                self.minimal = True  # an optional field was rejected: one retry with the bare body
                return self.chat(messages, max_tokens, deadline)
            raise LLMUnavailable(f"HTTP {exc.code}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError,
                ValueError) as exc:
            raise LLMUnavailable(f"{type(exc).__name__}") from None
        try:
            msg = payload["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip() and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return str(content)
        except (KeyError, IndexError, TypeError):
            raise LLMUnavailable("malformed completion payload") from None


def log(msg: str) -> None:
    print(f"[agenthon.llm] {msg}", file=sys.stderr)
