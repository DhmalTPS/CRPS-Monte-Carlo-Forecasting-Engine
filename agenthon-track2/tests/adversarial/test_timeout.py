import time

from agenthon.main import run
from agenthon.safety import Deadline, run_with_timeout


def test_run_with_timeout_returns_default():
    t = time.monotonic()
    assert run_with_timeout(lambda: time.sleep(5), 0.2, "dflt") == "dflt"
    assert time.monotonic() - t < 2
    assert run_with_timeout(lambda: 1 / 0, 1.0, "dflt") == "dflt"
    assert run_with_timeout(lambda: 42, 1.0) == 42


def test_deadline():
    d = Deadline(0.05)
    time.sleep(0.1)
    assert d.expired and d.left() < 0


class Slow:
    calls = 0
    available = True

    def chat(self, *a, **k):
        raise AssertionError("must not be called when the text budget is exhausted")


def test_no_llm_call_without_budget(unit, out_dir):
    path, _ = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                  overrides={"runtime": {"text_seconds": 1}}, client=Slow())
    assert path == "full"
