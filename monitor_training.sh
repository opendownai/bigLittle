#!/bin/bash
# Monitor training until completion or 7am

LOG_FILE="/Users/kuma/workspace/opendown/lottery-prediction/training_night.log"
CKPT_DIR="/Users/kuma/.openclaw/workspace/ane-training/training_dynamic"

echo "=== Training Monitor Started ==="
echo "Log: $LOG_FILE"
echo ""

# Wait for training to complete or timeout at 7am
while true; do
    # Check if training process is still running
    if ! ps -p 34906 > /dev/null 2>&1; then
        echo "$(date): Training process completed!"
        break
    fi
    
    # Check if log shows efficiency report (training done)
    if grep -q "Efficiency Report" "$LOG_FILE" 2>/dev/null; then
        echo "$(date): Training completed!"
        break
    fi
    
    # Check time - stop at 7am
    HOUR=$(date +%H)
    if [ "$HOUR" -ge 7 ]; then
        echo "$(date): Reached 7am, stopping monitor"
        break
    fi
    
    # Show progress every 5 minutes
    if [ -f "$LOG_FILE" ]; then
        LAST_LOSS=$(grep "^step" "$LOG_FILE" | tail -1 | grep -oP "loss=\K[0-9.]+" || echo "N/A")
        LAST_STEP=$(grep "^step" "$LOG_FILE" | tail -1 | grep -oP "step \K[0-9]+" || echo "0")
        echo "$(date): Step $LAST_STEP, Loss: $LAST_LOSS"
    fi
    
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=== Final Status ==="
if [ -f "$LOG_FILE" ]; then
    echo "Last 10 lines of log:"
    tail -10 "$LOG_FILE"
fi

# List best checkpoints
echo ""
echo "Best checkpoints:"
ls -la "$CKPT_DIR"/ane_lottery_best*.bin 2>/dev/null | sort -t'_' -k3 -n | head -5

echo ""
echo "Monitor done at $(date)"
