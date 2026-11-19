"""M2_statistical — ablation rung (overrides in offline/ablations/_ladder.py RUNGS['M2_statistical'])."""

import sys

from offline.ablations._ladder import run_rung

if __name__ == "__main__":
    run_rung("M2_statistical", jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
