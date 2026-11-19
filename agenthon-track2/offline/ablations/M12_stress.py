"""M12 — stress test: the full system on crisis windows only (GFC, taper, COVID, 2022)
and under adversarial panel perturbations (see tests/adversarial). A rung passes only if it never
does materially worse than M0 on the stress slice."""

import sys

from offline.ablations._ladder import run_rung
from offline.walk_forward.run import main as _wf  # noqa: F401

STRESS_WINDOWS = [("2008-06-01", "2009-06-30"), ("2013-04-01", "2013-12-31"),
                  ("2020-01-15", "2020-07-31"), ("2022-01-01", "2022-12-31")]

if __name__ == "__main__":
    run_rung("M11_full", jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    print("stress slice: python offline/walk_forward/stress.py M11_full")
