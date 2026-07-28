#!/usr/bin/env python3
"""Generate one reproducible lottery prediction from the selected method."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from models.ane_lottery_evaluator import (
    DEFAULT_DEPLOYMENT_CHECKPOINT,
    DEFAULT_DEPLOYMENT_MANIFEST,
    DEFAULT_TRAINING_MANIFEST,
    predict_next_from_files as predict_next_ane,
)
from models.lottery_trainer import (
    default_model_path,
    predict_next as predict_next_ranker,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "dlt_merged.json"


def load_data(path: Path, year_filter: int | None = None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = [
        {
            **draw,
            "date": draw.get("date") or draw.get("drawDate"),
        }
        for draw in data
    ]
    normalized.sort(key=lambda draw: draw["date"])
    if year_filter is not None:
        normalized = [
            draw for draw in normalized if int(draw["date"][:4]) >= year_filter
        ]
    if not normalized:
        raise ValueError("No historical draws match the requested filter")
    return normalized


def frequency_prediction(
    data: list[dict], rng: random.Random, recent_draws: int = 50
) -> dict:
    front_counter = {ball: 0 for ball in range(1, 36)}
    back_counter = {ball: 0 for ball in range(1, 13)}
    for draw in data[-recent_draws:]:
        for ball in draw["frontBalls"]:
            front_counter[ball] += 1
        for ball in draw["backBalls"]:
            back_counter[ball] += 1

    front_pool = sorted(front_counter, key=lambda ball: (-front_counter[ball], ball))[
        :10
    ]
    back_pool = sorted(back_counter, key=lambda ball: (-back_counter[ball], ball))[:6]
    return {
        "front": sorted(rng.sample(front_pool, 5)),
        "back": sorted(rng.sample(back_pool, 2)),
        "method": "frequency",
    }


def random_prediction(rng: random.Random) -> dict:
    return {
        "front": sorted(rng.sample(range(1, 36), 5)),
        "back": sorted(rng.sample(range(1, 13), 2)),
        "method": "random",
    }


def balanced_prediction(rng: random.Random) -> dict:
    odd = rng.sample([ball for ball in range(1, 36) if ball % 2], 3)
    even = rng.sample([ball for ball in range(1, 36) if not ball % 2], 2)
    return {
        "front": sorted(odd + even),
        "back": sorted(rng.sample(range(1, 13), 2)),
        "method": "balanced",
    }


def generate_predictions(
    *,
    data_path: Path,
    model_path: Path,
    ane_checkpoint_path: Path,
    ane_training_manifest_path: Path,
    ane_deployment_manifest_path: Path,
    method: str,
    year_filter: int | None,
    count: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if method in {"ane", "model"} and year_filter is not None:
        raise ValueError(
            "--year is not used by model inference; train the checkpoint "
            "with the intended data"
        )
    data = load_data(data_path, year_filter)
    latest = data[-1]
    rng = random.Random(seed)
    predictions: list[dict] = []

    if method in {"ane", "model"}:
        if count != 1:
            raise ValueError(
                f"{method} method currently emits exactly one prediction"
            )
        if method == "ane":
            prediction = predict_next_ane(
                data_path,
                ane_checkpoint_path,
                ane_training_manifest_path,
                ane_deployment_manifest_path,
            )
        else:
            prediction = predict_next_ranker(data_path, model_path)
        predictions.append(
            {
                "front": prediction["front"],
                "back": prediction["back"],
                "method": prediction["method"],
            }
        )
    else:
        generators = {
            "frequency": lambda: frequency_prediction(data, rng),
            "random": lambda: random_prediction(rng),
            "balanced": lambda: balanced_prediction(rng),
        }
        for _ in range(count):
            predictions.append(generators[method]())

    return predictions, latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument(
        "--ane-checkpoint",
        type=Path,
        default=DEFAULT_DEPLOYMENT_CHECKPOINT,
    )
    parser.add_argument(
        "--ane-training-manifest",
        type=Path,
        default=DEFAULT_TRAINING_MANIFEST,
    )
    parser.add_argument(
        "--ane-deployment-manifest",
        type=Path,
        default=DEFAULT_DEPLOYMENT_MANIFEST,
    )
    parser.add_argument(
        "--method",
        choices=("ane", "model", "frequency", "random", "balanced"),
        default="ane",
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, latest = generate_predictions(
        data_path=args.data,
        model_path=args.model,
        ane_checkpoint_path=args.ane_checkpoint,
        ane_training_manifest_path=args.ane_training_manifest,
        ane_deployment_manifest_path=args.ane_deployment_manifest,
        method=args.method,
        year_filter=args.year,
        count=args.count,
        seed=args.seed,
    )
    print(f"Loaded through {latest['issueNumber']} ({latest['date']})")
    for index, prediction in enumerate(predictions, 1):
        front = " ".join(f"{value:02d}" for value in prediction["front"])
        back = " ".join(f"{value:02d}" for value in prediction["back"])
        print(f"{index}. {front} + {back} ({prediction['method']})")


if __name__ == "__main__":
    main()
