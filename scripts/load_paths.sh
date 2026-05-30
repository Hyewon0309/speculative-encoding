#!/usr/bin/env bash
# Load a JSON config from configs/*.json and export every leaf as an env var.
# Usage:  source scripts/load_paths.sh configs/paths.json
# JSON values may reference other keys with ${VAR} (resolved in source order).

set -u
CONFIG_FILE="${1:-configs/paths.json}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[load_paths] Config not found: $CONFIG_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

# Python is used to (a) parse JSON, (b) keep insertion order, (c) resolve
# ${VAR} references against the current environment as each key is exported.
eval "$(python - "$CONFIG_FILE" <<'PY'
import json, os, re, sys, shlex
with open(sys.argv[1]) as f:
    data = json.load(f)
pat = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
for k, v in data.items():
    if k.startswith("_") or not isinstance(v, str):
        continue
    resolved = pat.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
    os.environ[k] = resolved
    if k == "PYTHONPATH_PREPEND":
        cur = os.environ.get("PYTHONPATH", "")
        new = resolved if not cur else f"{resolved}:{cur}"
        print(f"export PYTHONPATH={shlex.quote(new)}")
        continue
    print(f"export {k}={shlex.quote(resolved)}")
PY
)"
echo "[load_paths] Loaded $(basename "$CONFIG_FILE")"
