"""Adaptive reasoning policy: decides how much machinery to trust for THIS card."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..resources import load_artifact, select_vintage
from ..types import PolicyAction
from .actions import ACTIONS, action_index, action_key
from .bandit import SafePolicy
from .features import NAMES, context_vector


def choose_action(x: np.ndarray, asof: str, cfg: dict[str, Any]) -> PolicyAction:
    dflt = PolicyAction(gamma=float(cfg["decision"].get("gamma", 1.25)), text_strength=1.0,
                        lam=float(cfg["decision"].get("lambda", 0.2)))
    if not cfg.get("policy", {}).get("enabled", True):
        return dflt
    try:
        v = select_vintage(load_artifact("policy_vintages.json"), asof)
        if v is None:
            return dflt
        pol = SafePolicy.from_dict(v["policy"])
        if pol.arms[0].dim != len(x):
            return dflt
        a, info = pol.select(x, float(cfg["policy"].get("pessimism", 1.0)))
        if a == pol.default:
            return PolicyAction(dflt.gamma, 1.0, dflt.lam, "policy:default",
                                {k: float(v2) for k, v2 in info.items()})
        g, lam = ACTIONS[a]
        return PolicyAction(float(g), 1.0, float(lam), f"policy:{v.get('fit_cutoff')}",
                            {k: float(v2) for k, v2 in info.items()})
    except Exception as exc:
        print(f"[agenthon] policy unavailable ({type(exc).__name__}); default action", file=sys.stderr)
        return dflt


__all__ = ["ACTIONS", "NAMES", "SafePolicy", "action_index", "action_key", "choose_action", "context_vector"]
