"""M0_baseline — ablation rung (overrides in offline/ablations/_ladder.py RUNGS['M0_baseline'])."""

import sys

from offline.ablations._ladder import run_rung

if __name__ == "__main__":
    run_rung("M0_baseline", jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
