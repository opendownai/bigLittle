#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${1:-}"

if [[ -z "$LOG_PATH" ]]; then
    LOG_PATH="$(ls -t "$PROJECT_ROOT"/ane_training/lottery_ane_*.log 2>/dev/null | head -1 || true)"
fi
if [[ -z "$LOG_PATH" || ! -f "$LOG_PATH" ]]; then
    echo "No ANE training log found." >&2
    exit 1
fi

echo "Log: $LOG_PATH"
pgrep -fal 'training_dynamic/train|./train --(scratch|resume)' || true
tail -30 "$LOG_PATH"
