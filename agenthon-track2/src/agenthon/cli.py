"""The `forecast` verb:  forecast --panels /input/panels/ --text /input/text/ --asof D --out /output/forecast.parquet

Thread pools are pinned BEFORE numpy is imported (the runtime caps the container at 256 PIDs), and
HOME/caches point at /tmp because the root filesystem is read-only and uid 65534 has no home.
A leading literal 'forecast' argument is accepted too (ENTRYPOINT-style invocation).
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import pathlib

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "forecast":
        args = args[1:]
    p = argparse.ArgumentParser(prog="forecast", description="Agenthon 2026 Track 2 forecaster")
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", default=None)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--card", type=pathlib.Path, default=None)
    p.add_argument("--profile", default=None, help="config profile (development|final)")
    p.add_argument("--no-llm", action="store_true", help="never call the House model")
    a = p.parse_args(args)
    overrides = {"llm": {"enabled": False}} if a.no_llm else None
    from .main import run

    run(a.panels, a.text, a.asof, a.out, a.card, a.profile, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
