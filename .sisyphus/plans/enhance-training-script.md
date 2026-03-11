# 训练脚本增强计划

## 目标
修改 `start_training.sh`，添加以下功能：
1. 每 10000 步 Telegram 推送进度
2. 检测到扩散（loss 持续变差）或训练失败时自动停止

## 任务清单

- [x] 添加 `LAST_NOTIFIED_STEP` 变量和 `get_current_step()` 函数
- [x] 添加 `CONSECUTIVE_BAD_LOSS` 变量用于检测扩散
- [x] 在主循环中添加每 10000 步推送逻辑
- [x] 添加扩散检测逻辑（连续 N 次 loss 超过阈值则停止）
- [x] 添加训练失败自动停止逻辑
- [x] 更新 TRAINING.md 文档

## 详细修改

### 1. 新增变量和函数（在 `LAST_NAN_NOTIFY=0` 后）

```bash
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
```

### 2. 每 10000 步推送（在主循环 `sleep 10` 前）

```bash
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
```

### 3. 扩散检测（在主循环中）

```bash
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
```

## Final Verification Wave

- [x] F1: 脚本语法检查 `bash -n start_training.sh`
- [x] F2: 功能逻辑审查 - 确认所有新增功能正确实现
- [x] F3: TRAINING.md 更新确认