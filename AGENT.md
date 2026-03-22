# Lottery Prediction Agent 工作流程

## 重要更新 (2026-03-22)
- 模型训练只使用2025年至今的数据 (173条)
- 每次只预测1个号码

## 数据源
- 官方PDF: `https://pdf.sporttery.cn/33800/{期号}/{期号}.pdf`
- 本地数据: `data/dlt_merged.json` (已在 .gitignore 中)

## 完整工作流程

### 1. 获取最新开奖数据
```bash
python3 -c "
import requests, subprocess, re
from pathlib import Path
for issue in ['26030', '26031']:
    url = f'https://pdf.sporttery.cn/33800/{issue}/{issue}.pdf'
    # ... 下载并解析
"
```

### 2. 更新数据文件
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
"
```

### 3. 重新准备训练数据 (重要!)
```bash
python3 prepare_data.py --year 2025
```

### 4. 启动模型训练
```bash
cd $HOME/.openclaw/workspace/ane-training/training/training_dynamic
./train --resume --data lottery_train.bin
```

### 5. 生成预测 (1个号码)
```bash
source venv/bin/activate
python3 generate_predictions.py --year 2025 --count 1
```

### 6. 写入预测文件
```
pre/YYYY-MM-DD.txt
```

格式 (只输出1注):
```
Lottery Prediction
Generated on: 2026-03-23
Based on 173 historical draws (2025+)

 1. 02 09 14 18 26 + 05 12 (frequency)
```

### 7. 开奖后分析
```bash
python3 -c "
actual = {'frontBalls': [...], 'backBalls': [...]}
# ... 统计中奖
"
```

### 8. 提交 GitHub
```bash
git add -A
git commit -m "Add prediction for YYYY-MM-DD"
git push origin main
```

## 关键文件
- 数据: `data/dlt_merged.json`
- 预测: `pre/YYYY-MM-DD.txt`
- 分析: `analysis/YYYY-MM-DD_analysis.md`
- 模型: `$HOME/.openclaw/workspace/ane-training/training_training_dynamic/ane_lottery_*.bin`
- 训练数据: `data/lottery_train.bin`

## 当前状态 (2026-03-22)
- 数据: 2108条 (2025年至今173条)
- 最新开奖: 26029 (2026-03-21): 03 05 17 33 35 + 05 07
- 下期预测: 2026-03-23
- 等待数据: 26030+
