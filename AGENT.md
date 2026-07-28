# Lottery Prediction Agent 工作流程

## 原则

- 只使用经过来源交叉核验的真实、连续开奖数据。
- 当前数据范围固定为 2024、2025、2026 三个自然年，不按天或按期开奖滚动。
- 所有训练、验证和预测严格按时间顺序；任何目标期只能使用它之前的数据。
- 不因某一次留出结果较好就声称模型能够稳定预测随机开奖。
- 开奖前允许基于真实数据修订预测，并记录数据和模型哈希；开奖后不得换号。

## 1. 更新数据

```bash
python3 update_dlt_data.py
```

程序要求 GitHub 完整历史与中国体彩官方接口逐期完全一致后才写入：

- `data/dlt_merged.json`
- `data/dlt_merged.manifest.json`

只核验、不写文件：

```bash
python3 update_dlt_data.py --check-only
```

## 2. 时间留出评估 ANE

```bash
LOTTERY_HOLDOUT_DRAWS=72 ./start_training.sh scratch
python3 models/ane_lottery_evaluator.py \
  --checkpoint ane_training/snapshots/<run>/step_004000.bin \
  --split validation
python3 models/ane_lottery_evaluator.py \
  --checkpoint ane_training/snapshots/<run>/step_004000.bin \
  --split test
```

最后 72 期分成 36 期验证集和 36 期测试集。验证集用于选 checkpoint，
测试集只在选择完成后评估一次。

## 3. 使用全部三年数据训练部署模型

```bash
LOTTERY_HOLDOUT_DRAWS=0 \
LOTTERY_SNAPSHOT_DIR="$PWD/ane_training/deploy/three_calendar_years_$(date +%Y%m%d)" \
./start_training.sh scratch
```

将选定的最终 checkpoint 放在本地 `models/`，并保留其 SHA-256。
模型二进制和训练 token 文件是本机产物，不提交 Git。

## 4. 生成一注预测

```bash
python3 generate_predictions.py
```

默认使用 ANE。程序会核对数据 SHA、训练记录数、留出记录数和最新日期。
预测分别写入不可回写的 `predictions/<期号>.json` 和可读的
`pre/<期号>.txt`，同时记录数据范围、数据哈希、模型哈希和方法。

## 5. 开奖后评估

只在原预测文件或分析文件中追加真实开奖号码、前后区命中数和奖级，不覆盖原预测。
模型比较应同时报告时间留出结果、频率基线和随机理论期望。

## 6. 提交

先检查工作区，避免把用户原有或无关改动带入提交：

```bash
git status --short
git diff --check
git add <本次明确修改的文件>
git commit -m "Update audited data and ANE prediction"
git push origin main
```

## 7. 自动闭环

```bash
python3 automation/prediction_cycle.py --dry-run
python3 automation/manage_schedule.py install
python3 automation/manage_schedule.py status
```

调度任务每天 08:30 和 22:45 检查，但只在新期开奖且下一期预测不存在时训练。
处理顺序固定为“更新真实数据 → 分析旧预测 → 重训 → 生成新预测”。

LaunchAgent 会在一次闭环完整成功后提交并推送，但只暂存数据与来源清单、模型哈希、
期号预测记录和自动结果分析。暂存区中出现任何白名单外文件时必须停止，不能代替用户
提交其他改动。模型二进制、训练 token、ANE snapshot 和日志始终保持未跟踪。

## 当前状态（2026-07-28）

- 数据：386 期，24001 至 26084
- 数据 SHA-256：`b24321b45baa9beeb91579d56d7f44399b1c5a5b41f613490c5c2ce858079fcd`
- ANE：2 层、dim 64、85,312 参数、4,000 步
- 模型 SHA-256：`13bc8a48b7ddf23aa7dac73fd5a679e6a3e5c4e524738c70a62c3cdbe015c53d`
- 26085 期预测：`02 03 09 10 18 + 01 08`
- 新规则从 26014 期起改变奖级与奖金，不改变选号或开奖号码空间
