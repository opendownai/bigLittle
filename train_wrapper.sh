#!/bin/bash

TRAIN_DIR="$HOME/.openclaw/workspace/ane-training/training/training_dynamic"
CONTINUE_LOG="$HOME/.openclaw/workspace/ane-training/training/training_continue.log"
LOG_FILE="$HOME/.openclaw/workspace/ane-training/training/training_wrapper.log"
MAX_BEST_KEEP=5

BOT_TOKEN="8621293664:AAEv-ZD4zWsCVZ_uDvR_pcWWKNWu2QJCjyk"
CHAT_ID="5171247902"

send_telegram() {
    MSG="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d "chat_id=${CHAT_ID}" -d "text=$MSG" > /dev/null 2>&1
}

cleanup_old_checkpoints() {
    cd "$TRAIN_DIR" || return
    ls -1 ane_lottery_best_*.bin 2>/dev/null | grep -v nan | sort -t'_' -k3 -n | head -n -$MAX_BEST_KEEP | while read ckpt; do
        rm -f "$ckpt"
    done
}

get_best_loss() {
    ls -1 "$TRAIN_DIR"/ane_lottery_best_*.bin 2>/dev/null | grep -v nan | while read f; do
        echo "$f" | sed 's/.*_0\./0\./' | sed 's/\.bin//'
    done | sort -n | head -1
}

echo "=== Training Wrapper Started at $(date) ===" | tee -a "$LOG_FILE"

HISTORY_BEST=$(get_best_loss)
echo "Initial history best loss: $HISTORY_BEST" | tee -a "$LOG_FILE"
send_telegram "🚀 Training started! Historical best: loss=$HISTORY_BEST"

cd "$TRAIN_DIR"
nohup ./train --resume --data lottery_train.bin >> "$CONTINUE_LOG" 2>&1 &
sleep 3

if pgrep -f "train --resume" > /dev/null; then
    echo "$(date): Training started" | tee -a "$LOG_FILE"
else
    echo "$(date): Failed to start training" | tee -a "$LOG_FILE"
    exit 1
fi

while true; do
    if ! pgrep -f "train --resume" > /dev/null; then
        echo "$(date): Training process died, restarting..." | tee -a "$LOG_FILE"
        cd "$TRAIN_DIR"
        nohup ./train --resume --data lottery_train.bin >> "$CONTINUE_LOG" 2>&1 &
        sleep 3
        continue
    fi
    
    RECENT_LOSS=$(tail -50 "$CONTINUE_LOG" 2>/dev/null | grep "^step" | tail -1 | grep -oP "loss=\K[0-9.]+|nan" || echo "N/A")
    
    if [[ "$RECENT_LOSS" == "nan" ]] || [[ "$RECENT_LOSS" == "NaN" ]]; then
        echo "$(date): NaN detected, stopping training!" | tee -a "$LOG_FILE"
        pkill -f "train --resume"
        sleep 2
        
        BEST_CKPT=$(ls -1t "$TRAIN_DIR"/ane_lottery_best_*.bin 2>/dev/null | grep -v nan | head -1)
        
        if [[ -n "$BEST_CKPT" ]]; then
            MSG="🛑 Training stopped due to NaN! Best: $(basename $BEST_CKPT)"
            send_telegram "$MSG"
        else
            MSG="🛑 Training stopped due to NaN! No valid checkpoint."
            send_telegram "$MSG"
        fi
        
        echo "$(date): Wrapper finished" | tee -a "$LOG_FILE"
        break
    fi
    
    NEW_BEST=$(get_best_loss)
    
    if [[ -n "$NEW_BEST" ]] && (( $(echo "$NEW_BEST < $HISTORY_BEST" | bc -l) )); then
        IMPROVEMENT=$(echo "scale=4; ($HISTORY_BEST - $NEW_BEST) * 100 / $HISTORY_BEST" | bc)
        MSG="🎉 NEW ALL-TIME BEST! loss=$NEW_BEST (improved ${IMPROVEMENT}%)"
        echo "$(date): $MSG" | tee -a "$LOG_FILE"
        send_telegram "$MSG"
        
        HISTORY_BEST="$NEW_BEST"
        cleanup_old_checkpoints
    fi
    
    sleep 10
done
