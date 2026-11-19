"""M4_text — text rung, evaluated on the practice units (dated corpora exist only there).

Runs offline/walk_forward/practice_eval.py with the rung's text configuration and reports
delta_text = S(P0) - S(P1) per card (positive = text helped), by family.
"""

import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNG = "M4_text"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "offline/walk_forward/practice_eval.py"), "--rung", RUNG]
                   + sys.argv[1:], check=True)
