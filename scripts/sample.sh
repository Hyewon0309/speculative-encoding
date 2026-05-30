#!/usr/bin/env bash
# Run patch sampling on a directory of distilled patch features and write
# selected-index .npy files. Usage:
#   bash scripts/sample.sh configs/sampling/canonical_25pct.json \
#        --input_dir  /path/to/distilled_features/<dataset>/test \
#        --output_dir /path/to/indices/<exp>/<budget>
set -euo pipefail

CONFIG="${1:?Usage: bash scripts/sample.sh <sampler.json> [extra args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"

# Translate sampling config JSON into CLI flags.
ARGS=$(python - "$CONFIG" <<'PY'
import json, shlex, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
flags = []
for k, v in cfg.items():
    if k.startswith("_"):
        continue
    flag = f"--{k}"
    if isinstance(v, bool):
        if v: flags.append(flag)
    elif isinstance(v, (int, float)):
        flags += [flag, str(v)]
    else:
        flags += [flag, str(v)]
print(" ".join(shlex.quote(x) for x in flags))
PY
)

GPU_ID="${GPU_ID:-0}"
exec "${PYTHON:-python}" -m sampling \
  --device "cuda:${GPU_ID}" \
  --overwrite \
  $ARGS \
  "$@"
