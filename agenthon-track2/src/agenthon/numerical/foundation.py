"""Tier-2 foundation time-series model — DISABLED by rule, kept as an explicit, testable no-op.

The Track 2 artifact policy (docs/ARTIFACT-POLICY.md, revision 2026-09-21.1) requires *separate
organizer approval* for any pretrained neural checkpoint, "including Chronos and TimesFM weights".
We hold no such approval, so shipping one would put the whole submission at risk for a component
that the architecture review already classified as Conditional and never load-bearing
(Corrections C2/C3). This module therefore returns None, and the ensemble renormalises over the
remaining tiers, which is exactly the documented fallback for a missing/timeout Tier-2 vote.

If approval is ever granted, implement `forecast()` to return a per-step-point variance multiplier
(the only interface the ensemble accepts from any vote), gate it behind `gpu_available()` and a
hard timeout, and add it as ablation rung M12.
"""

from __future__ import annotations

from typing import Any

APPROVED = False


def gpu_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def forecast(*_args: Any, **_kwargs: Any) -> None:
    return None
