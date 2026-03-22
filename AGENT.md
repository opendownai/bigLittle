# Lottery Prediction Agent 工作流程

## 概述
大乐透 (超级大乐透) 预测系统，基于历史数据和神经网络模型进行预测。

## 数据源
- 官方PDF: `https://pdf.sporttery.cn/33800/{期号}/{期号}.pdf`
- 备用GitHub: `https://raw.githubusercontent.com/gudaoxuri/lottery_history/main/data/dlt.json`
- 本地数据: `data/dlt_merged.json` (注意: 已在 .gitignore 中)

## 完整工作流程

### 1. 获取最新开奖数据
```bash
# 检查最新期号 (当前是 26029)
# 尝试获取下期数据: 26030, 26031...
python3 -c "
import requests, subprocess, re
from pathlib import Path
for issue in ['26030', '26031']:
    url = f'https://pdf.sporttery.cn/33800/{issue}/{issue}.pdf'
    # ... 下载并解析
"
```

### 2. 更新数据文件 (如果获取到新数据)
```bash
python3 -c "
import json
from pathlib import Path
new_draw = {'issueNumber': '260XX', 'date': 'YYYY-MM-DD', 'frontBalls': [...], 'backBalls': [...]}
data_path = Path('data/dlt_merged.json')
with open(data_path) as f:
    data = json.load(f)
if not any(d['date'] == new_draw['date'] for d in data):
    data.append(new_draw)
    data.sort(key=lambda x: x['date'])
    with open(data_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('已更新')
"
```

### 3. 生成预测
```bash
# 激活虚拟环境
source venv/bin/activate

# 使用模型预测 (ANE Transformer)
python3 export_and_infer.py /Users/kuma/.openclaw/workspace/ane-training/training/training_dynamic/ane_lottery_best_510900_0.0881.bin

# 或使用频率分析
python3 -c "
import json, random
with open('data/dlt_merged.json') as f:
    data = json.load(f)
# ... 生成5组预测
"
```

### 4. 写入预测文件
```
pre/YYYY-MM-DD.txt
```

格式:
```markdown
# 大乐透预测 - 2026-03-23

## 最新开奖 (2026-03-21)
- 期号: 26029
- 前区: 03 05 17 33 35
- 后区: 05 07

## 预测号码 (2026-03-23)

### 1. 热门频率策略
- 前区: xx xx xx xx xx
- 后区: xx xx
...
```

### 5. 开奖后分析
如果需要分析预测表现:

```bash
# 1. 生成当时的100注预测 (使用开奖前数据)
python3 -c "
# 使用 不包含当期数据 的历史数据生成100注
"

# 2. 对比中奖情况
python3 -c "
actual = {'frontBalls': [...], 'backBalls': [...]}
# 统计中奖率、奖级分布、方法表现
"

# 3. 写入分析文件
analysis/YYYY-MM-DD_analysis.md
```

格式:
```markdown
# 大乐透 YYYY-MM-DD 开奖分析报告

## 实际开奖号码
前区: [...]
后区: [...]

## 预测表现统计
- 总预测数: 100
- 中奖预测数: XX
- 中奖率: XX.XX%

## 奖级分布
- X等奖: X 注
...

## 方法表现
- frequency: X/XX (XX.XX%)
- random: X/XX (XX.XX%)
- balanced: X/XX (XX.XX%)

## 详细中奖记录
...
```

### 6. 提交 GitHub
```bash
git add -A
git commit -m "Add prediction for YYYY-MM-DD (latest draw: XXXXX)"
git push origin main
```

## 重要文件位置
- 数据: `data/dlt_merged.json` (不提交)
- 预测: `pre/YYYY-MM-DD.txt`
- 分析: `analysis/YYYY-MM-DD_analysis.md`
- 模型: `/Users/kuma/.openclaw/workspace/ane-training/training/training_dynamic/ane_lottery_best_510900_0.0881.bin`
- 推理脚本: `export_and_infer.py`

## 当前状态 (2026-03-22)
- 最新开奖: 26029 (2026-03-21): 03 05 17 33 35 + 05 07
- 下期预测: 2026-03-23
- 等待数据: 26030+
