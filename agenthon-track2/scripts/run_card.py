"""Run one unit directory the way the harness does and print a short diagnostic.

    python scripts/run_card.py <unit-dir> [--out artifacts/forecasts/<unit>] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("unit", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    from agenthon.main import run

    out = a.out or ROOT / "artifacts" / "forecasts" / a.unit.name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    path, res = run(a.unit / "panels", a.unit / "text", None, out / "forecast.parquet",
                    overrides={"llm": {"enabled": False}} if a.no_llm else None)
    d = res.diagnostics if res else {}
    print(json.dumps({"unit": a.unit.name, "path": path, "gamma": d.get("gamma"), "lambda": d.get("lambda"),
                      "params": d.get("params"), "policy": d.get("policy"),
                      "text": {k: (d.get("text") or {}).get(k) for k in ("source", "gate", "kappa", "mechanism")},
                      "seconds": round(d.get("seconds", 0.0), 2), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
