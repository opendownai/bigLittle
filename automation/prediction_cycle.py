#!/usr/bin/env python3
"""Run one idempotent update, evaluation, ANE training, and prediction cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "automation" / "config.json"
DATA_PATH = PROJECT_ROOT / "data" / "dlt_merged.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "dlt_merged.manifest.json"
LOCK_PATH = PROJECT_ROOT / "data" / ".prediction_cycle.lock"
ISSUE_NAME = re.compile(r"^\d{5}$")

sys.path.insert(0, str(PROJECT_ROOT))
from models.ane_lottery_evaluator import (  # noqa: E402
    file_sha256,
    load_checkpoint,
    predict_next_from_files,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return True


def atomic_write_text(path: Path, content: str) -> bool:
    return atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> bool:
    content = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return atomic_write_text(path, content)


def run_command(
    arguments: list[str], *, environment: dict[str, str] | None = None
) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def publishable_paths(config: dict[str, Any]) -> list[Path]:
    paths = config["paths"]
    exact = [
        DATA_PATH,
        DATA_MANIFEST_PATH,
        project_path(paths["deploymentManifest"]),
    ]
    predictions_directory = project_path(paths["predictions"])
    human_directory = project_path(paths["humanPredictions"])
    analysis_directory = project_path(paths["analysis"])
    candidates = exact
    candidates.extend(sorted(predictions_directory.glob("*.json")))
    candidates.extend(
        path
        for path in sorted(human_directory.glob("*.txt"))
        if ISSUE_NAME.fullmatch(path.stem)
    )
    candidates.extend(sorted(analysis_directory.glob("*_result.json")))
    candidates.extend(sorted(analysis_directory.glob("*_result.md")))
    candidates.append(analysis_directory / "prediction_performance.json")
    return sorted({path for path in candidates if path.exists()})


def staged_paths() -> set[str]:
    output = git_output(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    )
    return {line for line in output.splitlines() if line}


def publish_cycle_changes(
    config: dict[str, Any],
    latest_issue: str,
    *,
    dry_run: bool,
) -> None:
    git_config = config.get("git", {})
    if not git_config.get("enabled", False):
        return
    candidates = publishable_paths(config)
    allowed = {
        str(path.relative_to(PROJECT_ROOT))
        for path in candidates
    }
    preexisting_staged = staged_paths()
    unexpected = preexisting_staged - allowed
    if unexpected:
        raise RuntimeError(
            "Refusing automated commit because unrelated paths are staged: "
            + ", ".join(sorted(unexpected))
        )

    if dry_run:
        print(
            "Would publish only: "
            + ", ".join(sorted(allowed))
        )
        return

    if candidates:
        run_command(
            [
                "git",
                "add",
                "--",
                *[
                    str(path.relative_to(PROJECT_ROOT))
                    for path in candidates
                ],
            ]
        )
    after_staging = staged_paths()
    unexpected = after_staging - allowed
    if unexpected:
        raise RuntimeError(
            "Automated staging escaped the allowlist: "
            + ", ".join(sorted(unexpected))
        )

    if after_staging:
        message = str(git_config["commitMessage"]).format(
            latestIssue=latest_issue
        )
        run_command(["git", "commit", "-m", message])
    else:
        print("No cycle changes to commit")

    if git_config.get("push", False):
        branch = git_output(["symbolic-ref", "--short", "HEAD"])
        remote = str(git_config.get("remote", "origin"))
        run_command(
            [
                "git",
                "push",
                remote,
                f"HEAD:refs/heads/{branch}",
            ]
        )


def update_data(calendar_years: int, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "update_dlt_data.py"),
        "--years",
        str(calendar_years),
    ]
    if dry_run:
        command.append("--check-only")
    run_command(command)


def choose_rule(
    issue: str, rules: dict[str, Any]
) -> dict[str, Any] | None:
    applicable = [
        version
        for version in rules["versions"]
        if int(issue) >= int(version["effectiveFromIssue"])
    ]
    return max(
        applicable,
        key=lambda version: int(version["effectiveFromIssue"]),
        default=None,
    )


def evaluate_prediction(
    record: dict[str, Any],
    actual: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    predicted_front = set(record["prediction"]["front"])
    predicted_back = set(record["prediction"]["back"])
    actual_front = set(int(value) for value in actual["frontBalls"])
    actual_back = set(int(value) for value in actual["backBalls"])
    front_hits = len(predicted_front.intersection(actual_front))
    back_hits = len(predicted_back.intersection(actual_back))
    condition = f"{front_hits}+{back_hits}"
    rule = choose_rule(str(actual["issueNumber"]), rules)
    tier = rule["tiers"].get(condition) if rule else None
    return {
        "schemaVersion": 1,
        "issue": str(actual["issueNumber"]),
        "drawDate": str(actual["date"]),
        "predictionRecord": record["recordPath"],
        "prediction": record["prediction"],
        "actual": {
            "front": sorted(actual_front),
            "back": sorted(actual_back),
        },
        "frontHits": front_hits,
        "backHits": back_hits,
        "totalHits": front_hits + back_hits,
        "condition": condition,
        "prizeTier": tier,
        "ruleVersion": rule["id"] if rule else None,
        "ruleSource": rule["source"] if rule else None,
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "actualDataSha256": file_sha256(DATA_PATH),
    }


def result_markdown(result: dict[str, Any]) -> str:
    prediction = result["prediction"]
    actual = result["actual"]
    predicted_numbers = " ".join(
        f"{value:02d}" for value in prediction["front"]
    )
    predicted_back = " ".join(
        f"{value:02d}" for value in prediction["back"]
    )
    actual_numbers = " ".join(f"{value:02d}" for value in actual["front"])
    actual_back = " ".join(f"{value:02d}" for value in actual["back"])
    tier = result["prizeTier"] or "未中奖"
    return (
        f"# 大乐透 {result['issue']} 期预测结果\n\n"
        f"- 开奖日期：{result['drawDate']}\n"
        f"- 原预测：`{predicted_numbers} + {predicted_back}`\n"
        f"- 实际号码：`{actual_numbers} + {actual_back}`\n"
        f"- 命中：前区 {result['frontHits']}，后区 {result['backHits']}，"
        f"合计 {result['totalHits']}\n"
        f"- 命中组合：`{result['condition']}`\n"
        f"- 结果：{tier}\n"
        f"- 规则版本：`{result['ruleVersion'] or '未配置'}`\n\n"
        "奖金、浮动奖金额和派奖金额以当期官方公告为准。\n"
    )


def rebuild_performance_summary(analysis_directory: Path) -> None:
    results = []
    for path in sorted(analysis_directory.glob("*_result.json")):
        value = read_json(path)
        if value.get("schemaVersion") == 1:
            results.append(value)
    if not results:
        return
    total_hits = sum(int(result["totalHits"]) for result in results)
    winning = sum(result["prizeTier"] is not None for result in results)
    tier_counts: dict[str, int] = {}
    for result in results:
        tier = result["prizeTier"] or "未中奖"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    summary = {
        "schemaVersion": 1,
        "evaluatedPredictions": len(results),
        "winningPredictions": winning,
        "averageTotalHits": total_hits / len(results),
        "tierCounts": tier_counts,
        "issues": [result["issue"] for result in results],
    }
    atomic_write_json(
        analysis_directory / "prediction_performance.json", summary
    )


def evaluate_pending(
    predictions_directory: Path,
    analysis_directory: Path,
    draws_by_issue: dict[str, dict[str, Any]],
    rules: dict[str, Any],
    dry_run: bool,
) -> int:
    evaluated = 0
    for path in sorted(predictions_directory.glob("*.json")):
        record = read_json(path)
        issue = str(record.get("issue", ""))
        actual = draws_by_issue.get(issue)
        result_path = analysis_directory / f"{issue}_result.json"
        if not issue or actual is None or result_path.exists():
            continue
        if dry_run:
            print(f"Would evaluate prediction for issue {issue}")
            evaluated += 1
            continue
        result = evaluate_prediction(record, actual, rules)
        atomic_write_json(result_path, result)
        atomic_write_text(
            analysis_directory / f"{issue}_result.md",
            result_markdown(result),
        )
        print(
            f"Evaluated {issue}: {result['condition']} "
            f"({result['prizeTier'] or '未中奖'})"
        )
        evaluated += 1
    if evaluated and not dry_run:
        rebuild_performance_summary(analysis_directory)
    return evaluated


def next_scheduled_date(
    latest_date: date, draw_weekdays: set[int]
) -> date:
    for offset in range(1, 15):
        candidate = latest_date + timedelta(days=offset)
        if candidate.weekday() in draw_weekdays:
            return candidate
    raise ValueError("No configured draw weekday found in the next 14 days")


def next_issue_number(latest_issue: str, scheduled_date: date) -> str:
    issue_year = int(latest_issue[:2])
    next_year = scheduled_date.year % 100
    if next_year != issue_year:
        return f"{next_year:02d}001"
    return f"{int(latest_issue) + 1:05d}"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_deployment(
    config: dict[str, Any],
    latest_issue: str,
) -> tuple[Path, Path, dict[str, Any]]:
    ane_config = config["ane"]
    paths = config["paths"]
    steps = int(ane_config["steps"])
    snapshot_directory = (
        PROJECT_ROOT / "ane_training" / "automation" / latest_issue
    )
    environment = os.environ.copy()
    environment.update(
        {
            "LOTTERY_HOLDOUT_DRAWS": str(
                int(ane_config["holdoutDraws"])
            ),
            "LOTTERY_TRAIN_STEPS": str(steps),
            "LOTTERY_SNAPSHOT_DIR": str(snapshot_directory),
        }
    )
    run_command(
        [str(PROJECT_ROOT / "start_training.sh"), "scratch"],
        environment=environment,
    )

    snapshot = snapshot_directory / f"step_{steps:06d}.bin"
    if not snapshot.exists():
        raise FileNotFoundError(f"Final ANE snapshot not found: {snapshot}")
    checkpoint_path = project_path(paths["checkpoint"])
    atomic_copy(snapshot, checkpoint_path)

    training_manifest_path = project_path(paths["trainingManifest"])
    deployment_manifest_path = project_path(paths["deploymentManifest"])
    training_manifest = read_json(training_manifest_path)
    checkpoint_config, _, _, _ = load_checkpoint(checkpoint_path)
    deployment_manifest = {
        "schemaVersion": 1,
        "sourceDataSha256": file_sha256(DATA_PATH),
        "trainingDataSha256": str(training_manifest["data_sha256"]),
        "checkpointSha256": file_sha256(checkpoint_path),
        "sourceDrawCount": int(training_manifest["source_draw_count"]),
        "trainingDrawCount": int(training_manifest["draw_count"]),
        "latestIssue": str(training_manifest["latest_issue"]),
        "latestDate": str(training_manifest["date_to"]),
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "config": {
            "step": checkpoint_config.step,
            "layers": checkpoint_config.layers,
            "dim": checkpoint_config.dim,
            "hidden": checkpoint_config.hidden,
            "heads": checkpoint_config.heads,
            "sequence": checkpoint_config.sequence,
            "vocab": checkpoint_config.vocab,
        },
    }
    atomic_write_json(deployment_manifest_path, deployment_manifest)
    return checkpoint_path, training_manifest_path, deployment_manifest


def prediction_text(record: dict[str, Any]) -> str:
    prediction = record["prediction"]
    front = " ".join(f"{value:02d}" for value in prediction["front"])
    back = " ".join(f"{value:02d}" for value in prediction["back"])
    return (
        "Lottery Prediction\n"
        f"Generated on: {record['generatedAt'][:10]}\n"
        f"For draw: {record['issue']}\n"
        f"Scheduled date estimate: {record['scheduledDateEstimate']} "
        "(休市日除外)\n"
        f"Based on data through: {record['input']['latestIssue']} "
        f"({record['input']['latestDate']})\n"
        f"Data SHA-256: {record['input']['dataSha256']}\n"
        f"Model SHA-256: {record['model']['checkpointSha256']}\n"
        f"Method: {prediction['method']}\n\n"
        f" 1. {front} + {back} ({prediction['method']})\n\n"
        "This record was generated before the target draw. After the draw, "
        "write a separate result analysis and do not replace these numbers.\n"
    )


def generate_next_prediction(
    config: dict[str, Any],
    latest_draw: dict[str, Any],
    scheduled_date: date,
    target_issue: str,
) -> dict[str, Any]:
    paths = config["paths"]
    checkpoint_path, training_manifest_path, deployment = train_deployment(
        config, str(latest_draw["issueNumber"])
    )
    deployment_manifest_path = project_path(paths["deploymentManifest"])
    prediction = predict_next_from_files(
        DATA_PATH,
        checkpoint_path,
        training_manifest_path,
        deployment_manifest_path,
    )
    data_manifest = read_json(DATA_MANIFEST_PATH)
    record = {
        "schemaVersion": 1,
        "recordPath": f"predictions/{target_issue}.json",
        "issue": target_issue,
        "scheduledDateEstimate": scheduled_date.isoformat(),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "input": {
            "latestIssue": str(latest_draw["issueNumber"]),
            "latestDate": str(latest_draw["date"]),
            "drawCount": int(data_manifest["dataset"]["records"]),
            "calendarYears": int(data_manifest["window"]["years"]),
            "dataSha256": str(data_manifest["dataset"]["sha256"]),
        },
        "model": {
            "checkpointSha256": deployment["checkpointSha256"],
            "trainingDataSha256": deployment["trainingDataSha256"],
            "config": deployment["config"],
        },
        "prediction": {
            "front": prediction["front"],
            "back": prediction["back"],
            "method": prediction["method"],
        },
    }
    predictions_directory = project_path(paths["predictions"])
    human_directory = project_path(paths["humanPredictions"])
    atomic_write_json(predictions_directory / f"{target_issue}.json", record)
    atomic_write_text(
        human_directory / f"{target_issue}.txt",
        prediction_text(record),
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify sources and report work without writing or training",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="use the current audited dataset without checking remote sources",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    rules = read_json(project_path(config["paths"]["rules"]))
    paths = config["paths"]
    predictions_directory = project_path(paths["predictions"])
    analysis_directory = project_path(paths["analysis"])
    predictions_directory.mkdir(parents=True, exist_ok=True)
    analysis_directory.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another prediction cycle is already running")
            return 0

        if not args.skip_update:
            update_data(int(config["calendarYears"]), args.dry_run)

        draws = read_json(DATA_PATH)
        draws_by_issue = {
            str(draw["issueNumber"]): draw for draw in draws
        }
        evaluate_pending(
            predictions_directory,
            analysis_directory,
            draws_by_issue,
            rules,
            args.dry_run,
        )

        latest_draw = draws[-1]
        scheduled_date = next_scheduled_date(
            date.fromisoformat(str(latest_draw["date"])),
            {int(value) for value in config["drawWeekdays"]},
        )
        target_issue = next_issue_number(
            str(latest_draw["issueNumber"]), scheduled_date
        )
        target_record = predictions_directory / f"{target_issue}.json"
        if target_record.exists():
            print(
                f"Prediction for issue {target_issue} already exists; "
                "no training required"
            )
            publish_cycle_changes(
                config,
                str(latest_draw["issueNumber"]),
                dry_run=args.dry_run,
            )
            return 0
        if args.dry_run:
            print(
                f"Would train on data through {latest_draw['issueNumber']} "
                f"and predict issue {target_issue}"
            )
            return 0

        record = generate_next_prediction(
            config,
            latest_draw,
            scheduled_date,
            target_issue,
        )
        front = " ".join(
            f"{value:02d}" for value in record["prediction"]["front"]
        )
        back = " ".join(
            f"{value:02d}" for value in record["prediction"]["back"]
        )
        print(f"Created prediction {target_issue}: {front} + {back}")
        publish_cycle_changes(
            config,
            str(latest_draw["issueNumber"]),
            dry_run=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
