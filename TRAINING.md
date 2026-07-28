# 模型训练

## 数据更新与核验

```bash
python3 update_dlt_data.py
```

更新程序先读取 GitHub 完整历史，再读取中国体彩官方接口，并要求 2,902 期
标准化记录逐期完全相等；另一份固定提交的官方接口快照用于二次交叉验证。
号码、日期、期号、重复项和同年度期号连续性均通过后，才原子写入本地文件。

模型数据采用最近三个完整或进行中的自然年。以当前截止日期计算，就是
2024-01-01 至 2026-07-27，共 386 期；不是最近 N 天或最近 N 期的滚动窗口。
`--years` 可以显式改变自然年数量。

## ANE 时间留出训练

默认配置位于 `models/lottery_config.h`：

- 2 layers，dim 64，hidden 128
- 4 attention / KV heads
- sequence 64，vocabulary 48
- 85,312 parameters
- 4,000 steps，前 200 步 warmup，每 500 步保存一次

token 流按开奖时间从旧到新排列：

- 前区：0–34
- 后区：35–46
- 每期开奖后的分隔符：47

先保留最后 72 期，其中前 36 期只用于 checkpoint 选择，后 36 期在选择完成后
只打开一次：

```bash
LOTTERY_HOLDOUT_DRAWS=72 ./start_training.sh scratch
python3 models/ane_lottery_evaluator.py \
  --checkpoint ane_training/snapshots/<run>/step_004000.bin \
  --split validation
python3 models/ane_lottery_evaluator.py \
  --checkpoint ane_training/snapshots/<run>/step_004000.bin \
  --split test
```

不要用测试集选择步数。多个更早时间截点可以用于稳定性复核，但最终测试集仍应隔离。

选定配置后，用全部 386 期重新从头训练部署 checkpoint：

```bash
LOTTERY_HOLDOUT_DRAWS=0 \
LOTTERY_SNAPSHOT_DIR="$PWD/ane_training/deploy/three_calendar_years_$(date +%Y%m%d)" \
./start_training.sh scratch
```

部署前使用全量 manifest 验证数据哈希，并生成下一期预测：

```bash
cp ane_training/deploy/<run>/step_004000.bin models/ane_lottery_deploy.bin
python3 models/ane_lottery_evaluator.py \
  --checkpoint models/ane_lottery_deploy.bin \
  --predict-next
python3 generate_predictions.py
```

`models/ane_lottery_deploy.manifest.json` 绑定源数据、训练 token 和 checkpoint
的 SHA-256。`generate_predictions.py` 会拒绝使用只训练了部分数据、存在留出集、
数据哈希过期、checkpoint 身份不符，或训练截止日期不等于最新开奖日期的模型。

## 训练参数

`start_training.sh` 支持这些环境变量：

- `LOTTERY_TRAIN_STEPS`
- `LOTTERY_TRAIN_LR`
- `LOTTERY_TRAIN_SEED`
- `LOTTERY_TRAIN_WARMUP`
- `LOTTERY_TRAIN_ACCUM`
- `LOTTERY_TRAIN_CLIP`
- `LOTTERY_TRAIN_WEIGHT_DECAY`
- `LOTTERY_TRAIN_MIN_LR_FRAC`
- `LOTTERY_TRAIN_SAVE_EVERY`
- `LOTTERY_HOLDOUT_DRAWS`
- `LOTTERY_DATA_PATH`
- `LOTTERY_SNAPSHOT_DIR`
- `ANE_SOURCE_DIR`

resume 模式只允许训练数据哈希与 checkpoint 旁的 manifest 完全一致：

```bash
./start_training.sh resume
```

新开奖加入或数据修订后，必须重新准备数据并从头训练，不能在不同数据上强行续训。

## 候选排序器

候选排序器仍保留作比较实验：

```bash
python3 models/lottery_trainer.py --train-ensemble 5 --predict
python3 generate_predictions.py --method model
```

当前独立测试结果没有超过基线，因此默认预测不使用它。

## 自动闭环

`automation/prediction_cycle.py` 按下面的固定顺序运行：

```text
来源核验和数据更新
  -> 评价已开奖的旧预测
  -> 判断下一期预测是否已经存在
  -> 全量三年数据从头训练 ANE
  -> 更新本机部署模型与哈希清单
  -> 写入下一期结构化记录和可读文本
  -> 白名单提交并推送本期可审计变更
```

预测记录以期号为主键。预计开奖日期仅用于展示，即使出现休市或临时调期，
实际评价仍按官方数据中的期号匹配，不会错期。

调度配置位于 `automation/config.json`，当前每天 08:30 和 22:45 检查。
任务是幂等的：无新开奖时不会重复训练；官方接口与 GitHub 数据尚未一致时会失败
并保留旧数据，等下一次调度重试。

Git 自动提交配置位于 `automation/config.json`。提交前会检查现有暂存区，只允许
审计数据、模型哈希清单、结构化预测、期号预测文本和自动结果分析。任何其他已暂存
文件都会令任务停止；模型、token、snapshot 和日志不在白名单内，也受
`.gitignore` 排除。

```bash
python3 automation/prediction_cycle.py --dry-run
python3 automation/manage_schedule.py install
python3 automation/manage_schedule.py status
python3 automation/manage_schedule.py uninstall
```

`uninstall` 只移除当前项目的 LaunchAgent，不删除预测、分析、模型或日志。
