#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANE_ROOT="${ANE_SOURCE_DIR:-$PROJECT_ROOT/ane_training/upstream}"
DYNAMIC_DIR="$ANE_ROOT/training/training_dynamic"
DATA_PATH="${LOTTERY_DATA_PATH:-$PROJECT_ROOT/data/lottery_train.bin}"
DATA_MANIFEST="$DATA_PATH.manifest.json"
CHECKPOINT_MANIFEST="$DYNAMIC_DIR/ane_lottery_dyn_ckpt.data.json"
STEPS="${LOTTERY_TRAIN_STEPS:-4000}"
LEARNING_RATE="${LOTTERY_TRAIN_LR:-3e-4}"
SEED="${LOTTERY_TRAIN_SEED:-20260728}"
WARMUP="${LOTTERY_TRAIN_WARMUP:-200}"
ACCUM_STEPS="${LOTTERY_TRAIN_ACCUM:-10}"
GRAD_CLIP="${LOTTERY_TRAIN_CLIP:-1.0}"
WEIGHT_DECAY="${LOTTERY_TRAIN_WEIGHT_DECAY:-0.01}"
MIN_LR_FRAC="${LOTTERY_TRAIN_MIN_LR_FRAC:-0}"
SAVE_EVERY="${LOTTERY_TRAIN_SAVE_EVERY:-500}"
HOLDOUT_DRAWS="${LOTTERY_HOLDOUT_DRAWS:-72}"
MODE="${1:-scratch}"

if [[ "$MODE" != "scratch" && "$MODE" != "resume" ]]; then
    echo "Usage: $0 [scratch|resume] [additional train arguments]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ ! -f "$DYNAMIC_DIR/train.m" ]]; then
    echo "ANE source not found at $DYNAMIC_DIR" >&2
    exit 1
fi

if [[ "$MODE" == "scratch" ]]; then
    python3 "$PROJECT_ROOT/prepare_data.py" \
        --output "$DATA_PATH" \
        --holdout-draws "$HOLDOUT_DRAWS"
    cp "$DATA_MANIFEST" "$CHECKPOINT_MANIFEST"
else
    if [[ ! -f "$DATA_PATH" || ! -f "$CHECKPOINT_MANIFEST" ]]; then
        echo "Resume data or checkpoint manifest is missing." >&2
        exit 1
    fi
    EXPECTED_SHA="$(
        python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_sha256"])' \
            "$CHECKPOINT_MANIFEST"
    )"
    ACTUAL_SHA="$(shasum -a 256 "$DATA_PATH" | awk '{print $1}')"
    if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
        echo "Refusing resume: checkpoint and training data hashes differ." >&2
        exit 1
    fi
fi

cp "$PROJECT_ROOT/models/lottery_config.h" "$DYNAMIC_DIR/models/lottery.h"
make -C "$DYNAMIC_DIR" MODEL=lottery

mkdir -p "$PROJECT_ROOT/ane_training"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$PROJECT_ROOT/ane_training/lottery_ane_$RUN_ID.log"
SNAPSHOT_DIR="${LOTTERY_SNAPSHOT_DIR:-$PROJECT_ROOT/ane_training/snapshots/$RUN_ID}"
mkdir -p "$SNAPSHOT_DIR"
SNAPSHOT_DIR="$(cd "$SNAPSHOT_DIR" && pwd)"

echo "Mode: $MODE"
echo "Data: $DATA_PATH"
echo "Log:  $LOG_PATH"
echo "Snapshots: $SNAPSHOT_DIR"

(
    cd "$DYNAMIC_DIR"
    ./train "--$MODE" \
        --data "$DATA_PATH" \
        --steps "$STEPS" \
        --lr "$LEARNING_RATE" \
        --warmup "$WARMUP" \
        --accum "$ACCUM_STEPS" \
        --clip "$GRAD_CLIP" \
        --wd "$WEIGHT_DECAY" \
        --min-lr-frac "$MIN_LR_FRAC" \
        --save-every "$SAVE_EVERY" \
        --snapshot-dir "$SNAPSHOT_DIR" \
        --seed "$SEED" \
        "$@"
) 2>&1 | tee "$LOG_PATH"
