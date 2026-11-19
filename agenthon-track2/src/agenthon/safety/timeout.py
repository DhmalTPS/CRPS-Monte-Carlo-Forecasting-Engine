"""Wall-clock budgeting. The binding constraint is the phase-shared clock (~400-860 s/unit on
average, 1800 s hard per unit), so every optional stage runs against a Deadline and is skipped,
not waited for, when the budget is gone."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class Deadline:
    def __init__(self, seconds: float):
        self.t0 = time.monotonic()
        self.end = self.t0 + float(seconds)

    def left(self) -> float:
        return self.end - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def sub(self, seconds: float) -> float:
        """Absolute monotonic deadline for a stage: min(now + seconds, overall end)."""
        return min(time.monotonic() + float(seconds), self.end)

    @property
    def expired(self) -> bool:
        return self.left() <= 0


def run_with_timeout(fn: Callable[[], Any], seconds: float, default: Any = None) -> Any:
    """Run fn in a daemon thread; return `default` if it does not finish in time.

    The thread is abandoned (Python cannot kill it), which is acceptable for pure-numpy stages and
    for network calls that also carry their own socket timeout.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller as default
            box["e"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(max(float(seconds), 0.0))
    if th.is_alive() or "e" in box:
        return default
    return box.get("v", default)
