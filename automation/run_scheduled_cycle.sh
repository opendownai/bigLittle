#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"

if [[ ! -x "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c \
    'import torch; import numpy; import requests' >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" -c 'import torch; import numpy; import requests' >/dev/null

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$PROJECT_ROOT/automation/prediction_cycle.py"
