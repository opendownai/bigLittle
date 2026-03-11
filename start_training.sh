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
    local total=$(ls -1 ane_lottery_best_*.bin 2>/dev/null | grep -v nan | wc -l | tr -d ' ')
    if [[ $total -gt $MAX_BEST_KEEP ]]; then
        local to_remove=$((total - MAX_BEST_KEEP))
        ls -1 ane_lottery_best_*.bin 2>/dev/null | grep -v nan | while read f; do
            loss=$(basename "$f" | sed -E 's/.*_([0-9]+\.[0-9]+)\.bin/\1/')
            echo "$loss $f"
        done | sort -n | head -n $to_remove | while read loss ckpt; do
            rm -f "$ckpt"
            echo "$(date): Removed checkpoint: $ckpt (loss=$loss)" >> "$LOG_FILE"
        done
    fi
}

get_best_loss() {
    ls -1 "$TRAIN_DIR"/ane_lottery_best_*.bin 2>/dev/null | grep -v nan | while read f; do
        basename "$f" | sed -E 's/.*_([0-9]+\.[0-9]+)\.bin/\1/'
    done | sort -n | head -1
}

get_best_ckpt() {
    cd "$TRAIN_DIR" || return
    ls -1 ane_lottery_best_*.bin 2>/dev/null | grep -v nan | while read f; do
        loss=$(basename "$f" | sed -E 's/.*_([0-9]+\.[0-9]+)\.bin/\1/')
        echo "$loss $f"
    done | sort -n | head -1 | awk -v dir="$TRAIN_DIR" '{print dir "/" $2}'
}

reload_best_checkpoint() {
    BEST_CKPT=$(get_best_ckpt)
    if [[ -n "$BEST_CKPT" ]]; then
        cp "$BEST_CKPT" "$TRAIN_DIR/ane_lottery_dyn_ckpt.bin"
        echo "$(date): Reloaded best checkpoint: $BEST_CKPT" | tee -a "$LOG_FILE"
    fi
}

pkill -f "train --resume" 2>/dev/null
pkill -f "train_wrapper" 2>/dev/null
sleep 1

reload_best_checkpoint

echo "=== Training Wrapper Started at $(date) ===" | tee -a "$LOG_FILE"

HISTORY_BEST=$(get_best_loss)
echo "Initial history best loss: $HISTORY_BEST" | tee -a "$LOG_FILE"
send_telegram "🚀 Training started! Historical best: loss=$HISTORY_BEST"

NAN_RELOAD_COUNT=0
LAST_NAN_NOTIFY=0
LAST_NOTIFIED_STEP=0
CONSECUTIVE_BAD_LOSS=0
BAD_LOSS_THRESHOLD=0.5  # loss 超过此值视为异常
MAX_BAD_LOSS_COUNT=10   # 连续异常次数阈值

get_current_step() {
    tail -50 "$CONTINUE_LOG" 2>/dev/null | grep "^step" | tail -1 | grep -oP "^step \K[0-9]+" || echo "0"
}

get_recent_loss() {
    tail -50 "$CONTINUE_LOG" 2>/dev/null | grep "^step" | tail -1 | grep -oP "loss=\K[0-9.]+|nan" || echo "N/A"
}

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
        # 检查最近是否有 NaN
        if tail -100 "$CONTINUE_LOG" 2>/dev/null | grep -q "loss=nan\|NaN\|!!! NaN"; then
            echo "$(date): NaN detected in log, reloading best checkpoint..." | tee -a "$LOG_FILE"
            reload_best_checkpoint
            sleep 2
        fi
        
        echo "$(date): Training process died, restarting..." | tee -a "$LOG_FILE"
        cd "$TRAIN_DIR"
        nohup ./train --resume --data lottery_train.bin >> "$CONTINUE_LOG" 2>&1 &
        sleep 3
        continue
    fi
    
    RECENT_LOSS=$(tail -50 "$CONTINUE_LOG" 2>/dev/null | grep "^step" | tail -1 | grep -oP "loss=\K[0-9.]+|nan" || echo "N/A")
    
    if [[ "$RECENT_LOSS" == "nan" ]] || [[ "$RECENT_LOSS" == "NaN" ]]; then
        echo "$(date): NaN detected in running process, reloading best checkpoint!" | tee -a "$LOG_FILE"
        pkill -f "train --resume"
        sleep 2
        
        reload_best_checkpoint
        
        ((NAN_RELOAD_COUNT++))
        CURRENT_TIME=$(date +%s)
        
        if [[ $NAN_RELOAD_COUNT -le 1 ]] || [[ $(($CURRENT_TIME - LAST_NAN_NOTIFY)) -gt 300 ]]; then
            BEST_CKPT=$(get_best_ckpt)
            MSG="🛑 NaN detected! Reloaded: $(basename $BEST_CKPT)"
            send_telegram "$MSG"
            LAST_NAN_NOTIFY=$CURRENT_TIME
        fi
        
        HISTORY_BEST=$(get_best_loss)
        echo "$(date): Reset history best to: $HISTORY_BEST" | tee -a "$LOG_FILE"
        
        sleep 5
        continue
    fi
    
    NEW_BEST=$(get_best_loss)
    
    if [[ -n "$NEW_BEST" ]] && [[ -n "$HISTORY_BEST" ]] && (( $(echo "$NEW_BEST < $HISTORY_BEST" | bc -l) )); then
        IMPROVEMENT=$(echo "scale=4; ($HISTORY_BEST - $NEW_BEST) * 100 / $HISTORY_BEST" | bc)
        MSG="🎉 NEW ALL-TIME BEST! loss=$NEW_BEST (improved ${IMPROVEMENT}%)"
        echo "$(date): $MSG" | tee -a "$LOG_FILE"
        send_telegram "$MSG"
        
        HISTORY_BEST="$NEW_BEST"
        cleanup_old_checkpoints
    fi
    
    # 每 10000 步通知
    CURRENT_STEP=$(get_current_step)
    MILESTONE=$((CURRENT_STEP / 10000 * 10000))
    if [[ $CURRENT_STEP -ge 10000 ]] && [[ $MILESTONE -gt $LAST_NOTIFIED_STEP ]]; then
        RECENT_LOSS=$(get_recent_loss)
        MSG="📊 Step $MILESTONE | loss=$RECENT_LOSS | best=$HISTORY_BEST"
        send_telegram "$MSG"
        LAST_NOTIFIED_STEP=$MILESTONE
        echo "$(date): Notified at step $MILESTONE" | tee -a "$LOG_FILE"
    fi
    
    # 扩散检测：loss 持续过高则停止
    RECENT_LOSS=$(get_recent_loss)
    if [[ "$RECENT_LOSS" != "nan" ]] && [[ "$RECENT_LOSS" != "NaN" ]] && [[ "$RECENT_LOSS" != "N/A" ]]; then
        if (( $(echo "$RECENT_LOSS > $BAD_LOSS_THRESHOLD" | bc -l) )); then
            ((CONSECUTIVE_BAD_LOSS++))
            if [[ $CONSECUTIVE_BAD_LOSS -ge $MAX_BAD_LOSS_COUNT ]]; then
                MSG="🚨 Training diverging! loss=$RECENT_LOSS > $BAD_LOSS_THRESHOLD for $MAX_BAD_LOSS_COUNT steps. Stopping."
                send_telegram "$MSG"
                echo "$(date): $MSG" | tee -a "$LOG_FILE"
                pkill -f "train --resume"
                exit 1
            fi
        else
            CONSECUTIVE_BAD_LOSS=0
        fi
    fi
    
    sleep 10
done
