#!/usr/bin/env python3
"""
Complete lottery automation system with retry mechanism
- Primary: Official PDF source (most reliable)
- Secondary: GitHub data source
- Tertiary: Manual input fallback
- Automatic analysis and next prediction
- Retry mechanism: every 3 minutes, max 10 attempts
"""

import json
import requests
import subprocess
import re
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

PRIZE_AMOUNTS = {
    "一等奖": 9889639,
    "二等奖": 87504,
    "三等奖": 5000,
    "四等奖": 300,
    "五等奖": 150,
    "六等奖": 15,
    "七等奖": 5,
}


def update_weekly_summary(
    date_str, issue_number, total_bets, winning_count, cost, total_prize
):
    summary_path = Path(__file__).parent / "weekly_summary.md"

    net_profit = total_prize - cost
    win_rate = winning_count / total_bets * 100 if total_bets > 0 else 0

    new_row = f"| {date_str} | {issue_number} | {total_bets} | {cost} | {winning_count} | {total_prize} | {net_profit} | {win_rate:.2f}% |"

    if summary_path.exists():
        content = summary_path.read_text()
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("|") and "---" not in line and "期号" not in line:
                lines.insert(i, new_row)
                break
        content = "\n".join(lines)
    else:
        content = f"""# 大乐透每周损益汇总

| 日期 | 期号 | 投注注数 | 投入成本(元) | 中奖注数 | 总奖金(元) | 净损益(元) | 中奖率 |
|------|------|----------|--------------|----------|------------|------------|--------|
{new_row}

## 累计统计
- **总投入**: {cost}元
- **总回报**: {total_prize}元  
- **累计净损益**: {net_profit}元
- **平均中奖率**: {win_rate:.2f}%"""

    summary_path.write_text(content)


def get_latest_official_pdf_data():
    """Get latest lottery data from official PDF (primary source)"""
    try:
        # Try current issue numbers
        current_issues = ["26024", "26023", "26022", "26021", "26020"]

        for issue in current_issues:
            url = f"https://pdf.sporttery.cn/33800/{issue}/{issue}.pdf"

            # Download PDF
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue

            # Save temporarily
            pdf_path = Path(f"/tmp/lottery_{issue}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            # Extract text using pdftotext
            result = subprocess.run(
                ["pdftotext", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up
            pdf_path.unlink()

            if result.returncode == 0:
                text = result.stdout

                # Extract date
                date = None
                date_match = re.search(
                    r"开奖日期[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日", text
                )
                if date_match:
                    date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

                # Extract winning numbers
                front_balls = []
                back_balls = []
                winning_section = re.search(
                    r"本期开奖号码[：:]?\s*([\d\s]+)", text, re.DOTALL
                )
                if winning_section:
                    numbers = re.findall(r"\d+", winning_section.group(1))
                    if len(numbers) >= 7:
                        front_balls = sorted([int(x) for x in numbers[:5]])
                        back_balls = sorted([int(x) for x in numbers[5:7]])

                if front_balls and back_balls:
                    return {
                        "issueNumber": issue,
                        "date": date,
                        "frontBalls": front_balls,
                        "backBalls": back_balls,
                    }

    except Exception as e:
        print(f"Official PDF parsing failed: {e}")

    return None


def get_latest_github_data():
    """Get latest lottery data from GitHub (secondary source)"""
    try:
        url = "https://raw.githubusercontent.com/gudaoxuri/lottery_history/main/data/dlt.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                latest = data[0]
                return {
                    "issueNumber": latest["issueNumber"],
                    "date": latest["drawDate"],
                    "frontBalls": sorted(latest["frontBalls"]),
                    "backBalls": sorted(latest["backBalls"]),
                }
    except Exception as e:
        print(f"GitHub fetch failed: {e}")
    return None


def manual_input_fallback():
    """Manual input when automatic sources fail"""
    print("=== 手动输入大乐透开奖数据 ===")

    # Get today's date as default
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"开奖日期 (默认 {today}): ", end="")
    date_input = input().strip()
    draw_date = date_input if date_input else today

    # Get front balls
    print("前区号码 (5个数字，空格分隔): ", end="")
    front_input = input().strip()
    front_balls = [int(x) for x in front_input.split()]

    # Get back balls
    print("后区号码 (2个数字，空格分隔): ", end="")
    back_input = input().strip()
    back_balls = [int(x) for x in back_input.split()]

    # Get issue number
    print("期号 (如 26024): ", end="")
    issue_input = input().strip()
    if not issue_input:
        # Generate based on date
        year = draw_date[:4][-2:]
        day_of_year = datetime.strptime(draw_date, "%Y-%m-%d").timetuple().tm_yday
        issue_number = f"{year}{day_of_year:03d}"
    else:
        issue_number = issue_input

    return {
        "issueNumber": issue_number,
        "date": draw_date,
        "frontBalls": sorted(front_balls),
        "backBalls": sorted(back_balls),
    }


def update_merged_dataset(new_draw):
    """Update merged dataset with new draw"""
    data_path = Path(__file__).parent / "data" / "dlt_merged.json"

    # Load existing data
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Check if this draw already exists
    existing = False
    for item in data:
        if (
            item.get("date") == new_draw["date"]
            or item.get("issueNumber") == new_draw["issueNumber"]
        ):
            existing = True
            break

    if not existing:
        data.append(new_draw)
        data.sort(key=lambda x: x["date"])
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Added new draw {new_draw['date']} to dataset")
        return True
    else:
        print(f"ℹ️  Draw {new_draw['date']} already exists")
        return False


def try_get_lottery_data_with_retry(max_attempts=10, base_retry_interval=180):
    """Try to get lottery data with retry mechanism starting from 21:50"""
    print(
        f"🔄 Starting data retrieval with {max_attempts} attempts, random intervals between 3-5 minutes..."
    )

    # Wait until 21:50 if current time is before that
    now = datetime.now()
    target_time = now.replace(hour=21, minute=50, second=0, microsecond=0)
    if now < target_time:
        wait_seconds = (target_time - now).total_seconds()
        print(f"⏳ Waiting until 21:50 ({wait_seconds:.0f} seconds)...")
        time.sleep(wait_seconds)

    for attempt in range(max_attempts):
        print(f"🎯 Attempt {attempt + 1}/{max_attempts}...")

        # Try official PDF first (most reliable)
        lottery_data = get_latest_official_pdf_data()

        if not lottery_data:
            print("⚠️  Official PDF source failed, trying GitHub...")
            lottery_data = get_latest_github_data()

        if lottery_data:
            print("✅ Successfully retrieved lottery data!")
            return lottery_data
        else:
            if attempt < max_attempts - 1:
                # Random interval between 3-5 minutes (180-300 seconds)
                retry_interval = random.randint(180, 300)
                print(
                    f"⏳ No data found. Waiting {retry_interval // 60} minutes and {retry_interval % 60} seconds before next attempt..."
                )
                time.sleep(retry_interval)
            else:
                print("❌ All attempts failed after maximum retries")

    return None


def main():
    """Main automation function with retry mechanism"""
    print("🔄 Starting lottery automation system with retry mechanism...")

    # Try to get data with retry
    lottery_data = try_get_lottery_data_with_retry(max_attempts=10, retry_interval=180)

    if not lottery_data:
        print("⚠️  All automatic sources failed after retries, using manual input")
        try:
            lottery_data = manual_input_fallback()
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled")
            return
        except Exception as e:
            print(f"❌ Manual input failed: {e}")
            return

    if update_merged_dataset(lottery_data):
        from post_draw_analyzer import (
            load_predictions,
            check_winnings,
            calculate_statistics,
            save_analysis_report,
            generate_next_prediction,
            save_next_predictions,
        )
        from datetime import datetime, timedelta

        today = lottery_data["date"]

        draw_date = datetime.strptime(today, "%Y-%m-%d")
        DLT_DRAW_DAYS = [0, 2, 5]
        days_to_wait = next(
            (
                i
                for i in range(1, 8)
                if (draw_date + timedelta(days=i)).weekday() in DLT_DRAW_DAYS
            ),
            1,
        )
        next_date = (draw_date + timedelta(days=days_to_wait)).strftime("%Y-%m-%d")

        predictions = load_predictions(today)
        if not predictions:
            print(f"⚠️  未找到 {today} 的预测文件")
            return

        results = check_winnings(predictions, lottery_data)
        stats = calculate_statistics(results)

        save_analysis_report(today, lottery_data, results, stats)

        next_predictions = generate_next_prediction()
        save_next_predictions(next_date, next_predictions)

        update_weekly_summary(
            today,
            lottery_data["issueNumber"],
            len(predictions),
            stats["winning_count"],
            stats["winning_count"] * 2,
            sum(
                r.get("prize_tier") and PRIZE_AMOUNTS.get(r["prize_tier"], 0)
                for r in results
            ),
        )

        print(f"✅ 分析完成！下次预测为 {next_date}")
    else:
        print("ℹ️  No new data to process")


if __name__ == "__main__":
    main()
