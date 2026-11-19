"""Regenerate configs/default.yaml from agenthon.config.DEFAULTS (the single source of truth)."""

import pathlib

import yaml

from agenthon.config import DEFAULTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
HDR = ("# Frozen built-in defaults = mirror of agenthon.config.DEFAULTS (PRIORS, no fitted values).\n"
       "# Fitted values live in src/agenthon/resources/*_vintages.json with their fit cutoffs.\n"
       "# tests/unit/test_config.py fails if this file and DEFAULTS diverge. Regenerate with:\n"
       "#   python scripts/sync_config.py\n")

if __name__ == "__main__":
    (ROOT / "configs/default.yaml").write_text(HDR + yaml.safe_dump(DEFAULTS, sort_keys=False), encoding="utf-8")
    print("configs/default.yaml regenerated")
