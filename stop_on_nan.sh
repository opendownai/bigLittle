#!/bin/bash

LOG_FILE="/Users/kuma/workspace/opendown/lottery-prediction/training_night.log"
PID=37797
CHECK_INTERVAL=10

echo "=== NaN Monitor Started ==="
echo "Monitoring: PID $PID"
echo "Log: $LOG_FILE"
echo ""

last_nan_check=0

while true; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "$(date): Training process completed!"
        break
    fi
    
    if [ -f "$LOG_FILE" ]; then
        last_loss=$(tail -20 "$LOG_FILE" | grep "^step" | tail -1 | grep -oP "loss=\K[0-9.]+|nan" || echo "N/A")
        
        if [[ "$last_loss" == "nan" ]]; then
            last_nan_check=$((last_nan_check + 1))
            
            if [ $last_nan_check -ge 3 ]; then
                echo "$(date): ⚠️ NaN detected! Last 3 checks all NaN. Stopping training..."
                kill -9 $PID
                echo "$(date): Process $PID killed."
                break
            else
                echo "$(date): ⚠️ NaN detected (check $last_nan_check/3), waiting..."
            fi
        else
            last_nan_check=0
            step_num=$(tail -20 "$LOG_FILE" | grep "^step" | tail -1 | grep -oP "step \K[0-9]+" || echo "0")
            echo "$(date): Step $step_num, Loss: $last_loss (OK)"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

echo "=== Monitor Done ==="
