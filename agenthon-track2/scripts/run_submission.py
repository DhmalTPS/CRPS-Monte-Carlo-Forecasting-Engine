"""Prepare an upload: fill the descriptor with a real image digest, validate, seal, and print the
pack command. Never touches the Team Key (the toolkit's `pack` prompts for it on a hidden prompt).

    python scripts/run_submission.py --image docker.io/<you>/agenthon-t2@sha256:<digest> \
        --phase dev [--no-house]

Steps it performs (all local, no upload):
  1. load submission/submission.template.json, set image/phase/competition_id (and models=[] with
     --no-house for a model-free run),
  2. validate with qfbench2_common.contracts (SubmissionDescriptor.from_mapping),
  3. reseal descriptor_digest with seal_descriptor_digest,
  4. write submission/submission.json and print:
       qfbench2 submission pack --descriptor submission/submission.json --team-number <N> --out submission.zip
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="registry/repository@sha256:<64 hex>")
    ap.add_argument("--phase", default="dev", choices=["dev", "final"])
    ap.add_argument("--no-house", action="store_true")
    ap.add_argument("--team-id", default=None, help="derived id (qfbench2 submission alias); pack checks it")
    a = ap.parse_args()
    m = re.match(r"^([^/]+)/(.+)@(sha256:[0-9a-f]{64})$", a.image)
    if not m:
        print("image must look like registry/repository@sha256:<64 hex>", file=sys.stderr)
        return 2
    d = json.loads((ROOT / "submission/submission.template.json").read_text(encoding="utf-8"))
    d["image"] = {"registry": m.group(1), "repository": m.group(2), "digest": m.group(3)}
    d["phase"] = a.phase
    d["competition_id"] = f"agenthon2026-forecasting-{a.phase}"
    if a.team_id:
        d["team_id"] = a.team_id
    if a.no_house:
        d["models"] = []
    try:
        from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest  # type: ignore

        d = seal_descriptor_digest(d)
        SubmissionDescriptor.from_mapping(d)
    except ImportError:
        print("qfbench2-common not installed: cannot seal/validate (install toolkit v2.4.4)", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"descriptor invalid: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    out = ROOT / "submission/submission.json"
    out.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}\nnext:\n  qfbench2 submission pack --descriptor {out} --team-number <N> --out submission.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
