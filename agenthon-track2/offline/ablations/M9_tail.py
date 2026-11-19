"""M9_tail — ablation rung (overrides in offline/ablations/_ladder.py RUNGS['M9_tail'])."""

import sys

from offline.ablations._ladder import run_rung

if __name__ == "__main__":
    run_rung("M9_tail", jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
