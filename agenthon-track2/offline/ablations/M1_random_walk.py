"""M1_random_walk — ablation rung (overrides in offline/ablations/_ladder.py RUNGS['M1_random_walk'])."""

import sys

from offline.ablations._ladder import run_rung

if __name__ == "__main__":
    run_rung("M1_random_walk", jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
