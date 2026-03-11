# 训练启动命令

## 快速启动

```bash
cd /Users/kuma/workspace/opendown/lottery-prediction
./start_training.sh
```

## 优化后的超参数 (2026-03-10)

| 参数 | 值 | 说明 |
|------|-----|------|
| total_steps | 100,000 | 训练总步数 |
| warmup_steps | 500 | 学习率预热步数 |
| max_lr | 3e-4 | 最大学习率 |
| weight_decay | 0.01 | 权重衰减 (从 0.1 降低) |
| min_lr_frac | 0.0 | 最小学习率比例 (余弦退火到 0) |
| grad_clip | 1.0 | 梯度裁剪 |

## 逻辑说明

1. **自动检测历史最佳**: 从训练目录找 loss 最低的 checkpoint（按 loss 排序，不是按时间）
2. **只在突破历史最佳时通知**: loss 比历史最佳更低时 Telegram 推送
3. **保留最佳 5 个**: 自动清理多余的旧 checkpoint
4. **NaN 自动恢复**: 检测到 loss=nan 自动停止 → 加载最佳 checkpoint → 继续训练
5. **启动时加载最佳**: 每次启动自动从最佳 checkpoint 开始
6. **每 10000 步推送**: 每 10000 步自动推送当前 loss 和最佳 loss
7. **扩散检测**: 连续 10 次 loss > 0.5 时自动停止训练并通知

## 当前状态

- 历史最佳: `ane_lottery_best_510900_0.0881.bin` (loss=0.0881)
- 训练中 PID: `pgrep -f "train --resume"`
- 日志: `~/.openclaw/workspace/ane-training/training/training_continue.log`
- Wrapper 日志: `~/.openclaw/workspace/ane-training/training/training_wrapper.log`

## 唤醒词

> "按现在的逻辑启动训练" 或 "继续训练"
