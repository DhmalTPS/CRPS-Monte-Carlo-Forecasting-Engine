#!/usr/bin/env bash
# Local end-to-end smoke of the image exactly as the harness runs it (offline, read-only, uid 65534).
#   bash submission/run.sh <unit-dir> [image]
set -euo pipefail
UNIT="${1:?unit directory}"
IMAGE="${2:-agenthon-t2:1.0.1}"
OUT="$(mktemp -d)"
chmod 777 "$OUT"
ASOF="$(python3 - "$UNIT" <<'PY'
import sys, tomllib, pathlib
print(tomllib.loads(pathlib.Path(sys.argv[1], "card.toml").read_text())["provenance"]["data_cutoff"])
PY
)"
docker run --rm --platform linux/amd64 --network=none --read-only --user 65534:65534 \
  --cap-drop=ALL --security-opt no-new-privileges --pids-limit 256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m --cpus=4 --memory=16g \
  -e QFBENCH_SEED=0 -e QFBENCH_NETWORK=none \
  -v "$(cd "$UNIT" && pwd)":/input:ro -v "$OUT":/output \
  "$IMAGE" forecast --panels /input/panels/ --text /input/text/ --asof "$ASOF" --out /output/forecast.parquet
ls -la "$OUT"
python3 scripts/validate_submission.py --card "$UNIT/card.toml" --out "$OUT"
