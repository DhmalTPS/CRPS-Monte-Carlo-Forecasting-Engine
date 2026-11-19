"""Wall-clock benchmark per stage over practice units (the binding budget is the phase clock).

    python scripts/benchmark.py --units <units-dir> [--limit 20]
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from agenthon.main import run

    units = [u for u in sorted(a.units.iterdir()) if (u / "card.toml").is_file()]
    if a.limit:
        units = units[: a.limit]
    secs = []
    for u in units:
        with tempfile.TemporaryDirectory() as td:
            t = time.perf_counter()
            run(u / "panels", u / "text", None, pathlib.Path(td) / "forecast.parquet",
                overrides={"llm": {"enabled": False}})
            secs.append(time.perf_counter() - t)
    secs.sort()
    print(f"{len(secs)} units: mean {statistics.mean(secs):.2f}s median {statistics.median(secs):.2f}s "
          f"p95 {secs[int(0.95 * (len(secs) - 1))]:.2f}s max {secs[-1]:.2f}s (numeric path; add <= 2 x "
          f"llm.timeout_seconds for House calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
