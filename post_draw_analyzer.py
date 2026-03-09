#!/usr/bin/env python3
"""
Post-draw analysis and next prediction generator
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta


def load_predictions(date_str):
    """Load predictions for a specific date"""
    pred_path = Path(__file__).parent / "pre" / f"{date_str}.txt"
    if not pred_path.exists():
        return None

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() and line[0].isdigit():
                parts = line.strip().split(".")
                if len(parts) >= 2:
                    pred_part = parts[1].strip()
                    # Extract numbers: "05 08 12 13 16 + 05 08 (frequency)"
                    if "+" in pred_part:
                        balls_part = pred_part.split("+")[0].strip()
                        back_part = pred_part.split("+")[1].split("(")[0].strip()
                        front_balls = [int(x) for x in balls_part.split()]
                        back_balls = [int(x) for x in back_part.split()]
                        method = pred_part.split("(")[1].rstrip(")")
                        predictions.append(
                            {"front": front_balls, "back": back_balls, "method": method}
                        )

    return predictions


def load_historical_data():
    """Load historical lottery data"""
    data_path = Path(__file__).parent / "data" / "dlt_merged.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_winnings(predictions, actual_draw):
    """Check winnings for each prediction against actual draw"""
    results = []
    actual_front = set(actual_draw["frontBalls"])
    actual_back = set(actual_draw["backBalls"])

    for i, pred in enumerate(predictions):
        pred_front = set(pred["front"])
        pred_back = set(pred["back"])

        front_match = len(pred_front & actual_front)
        back_match = len(pred_back & actual_back)

        # Prize tiers (大乐透奖级规则)
        prize_tier = None
        if front_match == 5 and back_match == 2:
            prize_tier = "一等奖"
        elif front_match == 5 and back_match == 1:
            prize_tier = "二等奖"
        elif front_match == 5 and back_match == 0:
            prize_tier = "三等奖"
        elif front_match == 4 and back_match == 2:
            prize_tier = "四等奖"
        elif front_match == 4 and back_match == 1:
            prize_tier = "五等奖"
        elif front_match == 3 and back_match == 2:
            prize_tier = "六等奖"
        elif front_match == 4 and back_match == 0:
            prize_tier = "七等奖"
        elif (front_match == 4 and back_match == 0) or (
            front_match == 3 and back_match == 2
        ):
            prize_tier = "五等奖"
        elif (front_match == 3 and back_match == 1) or (
            front_match == 2 and back_match == 2
        ):
            prize_tier = "六等奖"
        elif (
            (front_match == 3 and back_match == 0)
            or (front_match == 2 and back_match == 1)
            or (front_match == 1 and back_match == 2)
            or (front_match == 0 and back_match == 2)
        ):
            prize_tier = "七等奖"

        results.append(
            {
                "prediction_index": i + 1,
                "front_match": front_match,
                "back_match": back_match,
                "prize_tier": prize_tier,
                "method": pred["method"],
            }
        )

    return results


def calculate_statistics(results):
    """Calculate winning statistics"""
    total_predictions = len(results)
    winning_predictions = [r for r in results if r["prize_tier"] is not None]
    winning_count = len(winning_predictions)

    # Count by prize tier
    tier_counts = {}
    for result in winning_predictions:
        tier = result["prize_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Count by method
    method_stats = {}
    for result in results:
        method = result["method"]
        if method not in method_stats:
            method_stats[method] = {"total": 0, "wins": 0}
        method_stats[method]["total"] += 1
        if result["prize_tier"] is not None:
            method_stats[method]["wins"] += 1

    return {
        "total_predictions": total_predictions,
        "winning_count": winning_count,
        "win_rate": winning_count / total_predictions if total_predictions > 0 else 0,
        "tier_counts": tier_counts,
        "method_stats": method_stats,
    }


def generate_next_prediction():
    """Generate predictions for next draw"""
    # Load historical data
    data = load_historical_data()

    # Generate 100 predictions using the same logic as before
    predictions = []
    methods = ["frequency", "random", "balanced"]

    for i in range(100):
        method = methods[i % len(methods)]
        pred = generate_single_prediction(data, method)
        predictions.append(pred)

    return predictions


def generate_single_prediction(data, method="frequency"):
    """Generate single prediction based on method"""
    if method == "frequency":
        # Use frequency from last 100 draws
        front_counter = {}
        back_counter = {}

        for draw in data[-100:]:
            for ball in draw["frontBalls"]:
                front_counter[ball] = front_counter.get(ball, 0) + 1
            for ball in draw["backBalls"]:
                back_counter[ball] = back_counter.get(ball, 0) + 1

        front_balls = sorted(front_counter.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        front_selected = []
        while len(front_selected) < 5:
            candidates = [ball for ball, _ in front_balls if ball not in front_selected]
            if candidates:
                front_selected.append(random.choice(candidates))
            else:
                break

        back_balls = sorted(back_counter.items(), key=lambda x: x[1], reverse=True)[:6]
        back_selected = []
        while len(back_selected) < 2:
            candidates = [ball for ball, _ in back_balls if ball not in back_selected]
            if candidates:
                back_selected.append(random.choice(candidates))
            else:
                break

        return {
            "front": sorted(front_selected),
            "back": sorted(back_selected),
            "method": "frequency",
        }

    elif method == "random":
        front = random.sample(range(1, 36), 5)
        back = random.sample(range(1, 13), 2)
        return {"front": sorted(front), "back": sorted(back), "method": "random"}

    elif method == "balanced":
        front_odds = random.sample([i for i in range(1, 36) if i % 2 == 1], 3)
        front_evens = random.sample([i for i in range(1, 36) if i % 2 == 0], 2)
        front = sorted(front_odds + front_evens)
        back = random.sample(range(1, 13), 2)
        return {"front": sorted(front), "back": sorted(back), "method": "balanced"}


def save_analysis_report(date_str, actual_draw, results, stats):
    """Save analysis report to file"""
    report_path = Path(__file__).parent / "analysis" / f"{date_str}_analysis.md"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 大乐透 {date_str} 开奖分析报告\n\n")
        f.write(f"## 实际开奖号码\n")
        f.write(f"前区: {sorted(actual_draw['frontBalls'])}\n")
        f.write(f"后区: {sorted(actual_draw['backBalls'])}\n\n")

        f.write(f"## 预测表现统计\n")
        f.write(f"- 总预测数: {stats['total_predictions']}\n")
        f.write(f"- 中奖预测数: {stats['winning_count']}\n")
        f.write(f"- 中奖率: {stats['win_rate']:.2%}\n\n")

        if stats["tier_counts"]:
            f.write(f"## 奖级分布\n")
            for tier, count in sorted(stats["tier_counts"].items()):
                f.write(f"- {tier}: {count} 注\n")
            f.write("\n")

        f.write(f"## 方法表现\n")
        for method, m_stats in stats["method_stats"].items():
            win_rate = m_stats["wins"] / m_stats["total"] if m_stats["total"] > 0 else 0
            f.write(
                f"- {method}: {m_stats['wins']}/{m_stats['total']} ({win_rate:.2%})\n"
            )
        f.write("\n")

        f.write(f"## 详细中奖记录\n")
        winning_results = [r for r in results if r["prize_tier"] is not None]
        if winning_results:
            for result in winning_results[:10]:  # Show top 10
                f.write(
                    f"- 第{result['prediction_index']:3d}注: {result['front_match']}+{result['back_match']} ({result['prize_tier']}) - {result['method']}\n"
                )
        else:
            f.write("- 无中奖记录\n")


def update_historical_data(draw_date, front_balls, back_balls, issue_number):
    """Update historical data with new draw result"""
    data_path = Path(__file__).parent / "data" / "dlt_merged.json"

    # Load existing data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if this draw already exists
    existing = False
    for item in data:
        if item.get("date") == draw_date or item.get("issueNumber") == issue_number:
            existing = True
            break

    if not existing:
        # Add new draw
        new_draw = {
            "issueNumber": issue_number,
            "date": draw_date,
            "frontBalls": sorted(front_balls),
            "backBalls": sorted(back_balls),
        }
        data.append(new_draw)

        # Sort by date
        data.sort(key=lambda x: x["date"])

        # Save updated data
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已将 {draw_date} 的开奖数据添加到历史数据集")
        return True
    else:
        print(f"数据 {draw_date} 已存在，跳过添加")
        return False


def save_next_predictions(date_str, predictions):
    """Save next predictions to file"""
    pred_path = Path(__file__).parent / "pre" / f"{date_str}.txt"
    pred_path.parent.mkdir(exist_ok=True)

    with open(pred_path, "w", encoding="utf-8") as f:
        f.write(f"Lottery Predictions (100 sets)\n")
        f.write(f"Generated on: {date_str}\n")
        f.write(f"Based on {len(load_historical_data())} historical draws\n")
        f.write("=" * 50 + "\n\n")

        for i, pred in enumerate(predictions, 1):
            front_str = " ".join(f"{x:02d}" for x in pred["front"])
            back_str = " ".join(f"{x:02d}" for x in pred["back"])
            f.write(f"{i:3d}. {front_str} + {back_str} ({pred['method']})\n")


def main():
    # Get today's date (2026-03-09)
    today = "2026-03-09"
    next_date = "2026-03-11"  # Next Wednesday

    # Load today's predictions
    predictions = load_predictions(today)
    if not predictions:
        print(f"未找到 {today} 的预测文件")
        return

    # Use actual draw result from PDF
    actual_draw = {
        "frontBalls": [2, 4, 8, 10, 21],
        "backBalls": [9, 12],
        "drawDate": today,
    }
    issue_number = "26024"

    # Update historical data
    update_historical_data(
        today, actual_draw["frontBalls"], actual_draw["backBalls"], issue_number
    )

    # Analyze predictions
    results = check_winnings(predictions, actual_draw)
    stats = calculate_statistics(results)

    # Save analysis report
    save_analysis_report(today, actual_draw, results, stats)

    # Generate next predictions
    next_predictions = generate_next_prediction()
    save_next_predictions(next_date, next_predictions)

    print(f"\n分析完成！下次预测为 {next_date} (周三)")
    print(f"已生成 {next_date} 的100注预测")


if __name__ == "__main__":
    main()
