#!/bin/sh
# Optional ENTRYPOINT-style wrapper: accepts a leading `forecast` verb (the harness passes it) and
# guarantees exit 0 once any admissible output exists. The image itself uses no ENTRYPOINT; the
# `forecast` console script on PATH already accepts the verb-as-command invocation.
set -u
if [ "${1:-}" = "forecast" ]; then shift; fi
forecast "$@"
status=$?
# The Python side never exits non-zero after writing the fail-safe; this is belt and braces.
if [ $status -ne 0 ]; then
  echo "[agenthon] forecast exited $status" >&2
fi
exit $status
