"""The fixed ablation ladder (pre-registered: rungs and order fixed BEFORE looking at results).

Numeric rungs are evaluated on the synthetic walk-forward cards (offline/walk_forward/run.py);
text rungs (M4-M8) need a dated corpus and are evaluated on the practice units
(offline/walk_forward/practice_eval.py), where realised outcomes are reconstructed from the
published history. Each rung is a strict superset of the previous one.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

OFF_DECISION = {"decision": {"enabled": False}}
GAUSS = {"numerical": {"df": 1000.0}, "tails": {"in_unit_backtest": False}}
NO_CAL = {"calibration": {"enabled": False}}
NO_POLICY = {"policy": {"enabled": False}}

RUNGS: dict[str, dict] = {
    "M0_baseline": {"ablation": {"submit_m0": True}},
    "M1_random_walk": {**OFF_DECISION, **GAUSS, **NO_CAL, **NO_POLICY,
                       "numerical": {"df": 1000.0, "lr_omega": 0.0, "vol_persistence": 0.0,
                                     "vr_clip": [1.0, 1.0], "drift_shrink": 0.0, "zlb_enabled": False}},
    "M2_statistical": {**OFF_DECISION, **GAUSS, **NO_CAL, **NO_POLICY},
    "M3_ensemble": {**OFF_DECISION, **GAUSS, **NO_CAL, **NO_POLICY, "analogue": {"enabled": True}},
    "M9_tail": {**OFF_DECISION, **NO_CAL, **NO_POLICY},
    "M10_calibration": {**OFF_DECISION, **NO_POLICY},
    "M11_full": {},
}
TEXT_RUNGS = ("M4_text", "M5_gate", "M6_contradiction", "M7_distribution_update", "M8_dependence")


def run_rung(name: str, extra: dict | None = None, jobs: int = 6, limit: int = 0) -> None:
    ov = {**RUNGS[name], **(extra or {})}
    cmd = [sys.executable, str(ROOT / "offline/walk_forward/run.py"), "--name", name,
           "--overrides", json.dumps(ov), "--jobs", str(jobs)]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
