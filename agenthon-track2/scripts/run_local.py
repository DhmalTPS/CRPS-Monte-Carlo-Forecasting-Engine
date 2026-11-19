"""Run the agent on every practice unit (local smoke), mimicking the container layout.

    python scripts/run_local.py --units ../_reference/track2-forecasting-public/units \
        --outputs artifacts/forecasts [--only F3] [--no-llm] [--jobs 4]

Each unit's panels are passed as <unit>/panels (absent in the repo layout -> the loader falls back
to the unit root, exactly as a staged unit's /input/panels would be read). Writes a summary CSV
to artifacts/diagnostics/run_local.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def one(unit: pathlib.Path, outputs: pathlib.Path, no_llm: bool) -> dict:
    from agenthon.main import run

    od = outputs / unit.name
    od.mkdir(parents=True, exist_ok=True)
    for f in od.iterdir():
        f.unlink()
    t = time.time()
    path, res = run(unit / "panels", unit / "text", None, od / "forecast.parquet",
                    overrides={"llm": {"enabled": False}} if no_llm else None)
    d = res.diagnostics if res else {}
    return {"unit": unit.name, "path": path, "seconds": round(time.time() - t, 2),
            "gamma": d.get("gamma"), "gate": (d.get("text") or {}).get("gate"),
            "source": (d.get("text") or {}).get("source"), "df": d.get("df")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=pathlib.Path, required=True)
    ap.add_argument("--outputs", type=pathlib.Path, default=ROOT / "artifacts" / "forecasts")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    units = [u for u in sorted(a.units.iterdir()) if (u / "card.toml").is_file() and a.only in u.name]
    rows = []
    with cf.ProcessPoolExecutor(a.jobs) if a.jobs > 1 else cf.ThreadPoolExecutor(1) as ex:
        for r in ex.map(one, units, [a.outputs] * len(units), [a.no_llm] * len(units)):
            rows.append(r)
            print(r, flush=True)
    diag = ROOT / "artifacts" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / "run_local.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fails = [r for r in rows if r["path"] != "full"]
    print(f"{len(rows)} units, {len(fails)} not on the full path: {[r['unit'] for r in fails]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
