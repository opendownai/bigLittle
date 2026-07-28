#!/usr/bin/env python3
"""Train and run a small, time-aware lottery candidate ranker.

The model scores every valid front/back candidate independently from the same
set of recency features.  Sharing weights across candidates keeps the parameter
count appropriate for the amount of historical data and avoids learning a
spurious embedding for each ball number.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

FRONT_SIZE = 35
BACK_SIZE = 12
FRONT_PICK = 5
BACK_PICK = 2
DEFAULT_WINDOWS = (5, 10, 20, 50, 100)
DEFAULT_HALF_LIVES = (8, 32)
MAX_SEGMENT_GAP_DAYS = 45
FORMAT_VERSION = 3


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    front: tuple[int, ...]
    back: tuple[int, ...]


@dataclass
class Sample:
    issue: str
    date: str
    front_features: torch.Tensor
    front_target: torch.Tensor
    back_features: torch.Tensor
    back_target: torch.Tensor


class CandidateRanker(nn.Module):
    """A shared scorer applied to every candidate in one lottery zone."""

    def __init__(self, feature_count: int, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    return project_root() / "data" / "dlt_merged.json"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "lottery_model_v2.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_draws(path: Path) -> list[Draw]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    draws: list[Draw] = []
    seen_issues: set[str] = set()
    seen_dates: set[str] = set()

    for item in raw:
        issue = str(item["issueNumber"])
        date = str(item.get("date") or item.get("drawDate"))
        front = tuple(sorted(int(value) for value in item["frontBalls"]))
        back = tuple(sorted(int(value) for value in item["backBalls"]))

        if issue in seen_issues or date in seen_dates:
            raise ValueError(f"Duplicate draw: issue={issue}, date={date}")
        if len(front) != FRONT_PICK or len(set(front)) != FRONT_PICK:
            raise ValueError(f"Invalid front balls in issue {issue}: {front}")
        if len(back) != BACK_PICK or len(set(back)) != BACK_PICK:
            raise ValueError(f"Invalid back balls in issue {issue}: {back}")
        if not all(1 <= value <= FRONT_SIZE for value in front):
            raise ValueError(f"Front ball out of range in issue {issue}: {front}")
        if not all(1 <= value <= BACK_SIZE for value in back):
            raise ValueError(f"Back ball out of range in issue {issue}: {back}")

        seen_issues.add(issue)
        seen_dates.add(date)
        draws.append(Draw(issue=issue, date=date, front=front, back=back))

    draws.sort(key=lambda draw: draw.date)
    return draws


def split_contiguous_segments(
    draws: Sequence[Draw], max_gap_days: int = MAX_SEGMENT_GAP_DAYS
) -> list[list[Draw]]:
    segments: list[list[Draw]] = []
    current: list[Draw] = []

    for draw in draws:
        if current:
            previous = datetime.fromisoformat(current[-1].date)
            current_date = datetime.fromisoformat(draw.date)
            if (current_date - previous).days > max_gap_days:
                segments.append(current)
                current = []
        current.append(draw)

    if current:
        segments.append(current)
    return segments


def feature_names(
    windows: Sequence[int] = DEFAULT_WINDOWS,
    half_lives: Sequence[int] = DEFAULT_HALF_LIVES,
) -> list[str]:
    names = [f"rate_{window}" for window in windows]
    names.extend(f"ema_{half_life}" for half_life in half_lives)
    names.extend(["gap", "short_long_trend", "global_rate"])
    return names


def candidate_features(
    history: Sequence[tuple[int, ...]],
    candidate: int,
    windows: Sequence[int],
    half_lives: Sequence[int],
) -> list[float]:
    occurrences = [1.0 if candidate in draw else 0.0 for draw in history]
    rates: list[float] = []
    for window in windows:
        available = min(window, len(occurrences))
        rates.append(sum(occurrences[-available:]) / available)

    ema_rates: list[float] = []
    reversed_occurrences = list(reversed(occurrences))
    for half_life in half_lives:
        weights = [0.5 ** (age / half_life) for age in range(len(occurrences))]
        numerator = sum(value * weight for value, weight in zip(reversed_occurrences, weights))
        ema_rates.append(numerator / sum(weights))

    gap = next(
        (age for age, value in enumerate(reversed_occurrences) if value),
        len(occurrences),
    )
    normalized_gap = min(gap, max(windows)) / max(windows)
    short_long_trend = rates[1] - rates[-2]
    global_rate = sum(occurrences) / len(occurrences)
    return rates + ema_rates + [normalized_gap, short_long_trend, global_rate]


def zone_features(
    history: Sequence[tuple[int, ...]],
    zone_size: int,
    windows: Sequence[int],
    half_lives: Sequence[int],
) -> torch.Tensor:
    return torch.tensor(
        [
            candidate_features(history, candidate, windows, half_lives)
            for candidate in range(1, zone_size + 1)
        ],
        dtype=torch.float32,
    )


def multi_hot(values: Iterable[int], size: int) -> torch.Tensor:
    target = torch.zeros(size, dtype=torch.float32)
    for value in values:
        target[value - 1] = 1.0
    return target


def build_samples(
    draws: Sequence[Draw],
    min_history: int,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    half_lives: Sequence[int] = DEFAULT_HALF_LIVES,
) -> list[Sample]:
    samples: list[Sample] = []
    for segment in split_contiguous_segments(draws):
        for target_index in range(min_history, len(segment)):
            target = segment[target_index]
            history = segment[:target_index]
            front_history = [draw.front for draw in history]
            back_history = [draw.back for draw in history]
            samples.append(
                Sample(
                    issue=target.issue,
                    date=target.date,
                    front_features=zone_features(
                        front_history, FRONT_SIZE, windows, half_lives
                    ),
                    front_target=multi_hot(target.front, FRONT_SIZE),
                    back_features=zone_features(
                        back_history, BACK_SIZE, windows, half_lives
                    ),
                    back_target=multi_hot(target.back, BACK_SIZE),
                )
            )
    return samples


def stack_samples(
    samples: Sequence[Sample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("No samples available")
    return (
        torch.stack([sample.front_features for sample in samples]),
        torch.stack([sample.front_target for sample in samples]),
        torch.stack([sample.back_features for sample in samples]),
        torch.stack([sample.back_target for sample in samples]),
    )


def normalization(
    samples: Sequence[Sample],
) -> dict[str, dict[str, torch.Tensor]]:
    front_x, _, back_x, _ = stack_samples(samples)

    def stats(values: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = values.reshape(-1, values.shape[-1])
        return {
            "mean": flat.mean(dim=0),
            "std": flat.std(dim=0).clamp_min(1e-6),
        }

    return {"front": stats(front_x), "back": stats(back_x)}


def normalize_features(
    values: torch.Tensor, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    return (values - stats["mean"]) / stats["std"]


def model_states(
    front_model: CandidateRanker, back_model: CandidateRanker
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "front": copy.deepcopy(front_model.state_dict()),
        "back": copy.deepcopy(back_model.state_dict()),
    }


def restore_states(
    states: dict[str, dict[str, torch.Tensor]],
    front_model: CandidateRanker,
    back_model: CandidateRanker,
) -> None:
    front_model.load_state_dict(states["front"])
    back_model.load_state_dict(states["back"])


def batch_loss(
    front_model: CandidateRanker,
    back_model: CandidateRanker,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    stats: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    front_x, front_y, back_x, back_y = batch
    front_logits = front_model(normalize_features(front_x, stats["front"]))
    back_logits = back_model(normalize_features(back_x, stats["back"]))
    front_loss = nn.functional.binary_cross_entropy_with_logits(front_logits, front_y)
    back_loss = nn.functional.binary_cross_entropy_with_logits(back_logits, back_y)
    return front_loss + back_loss


@torch.no_grad()
def evaluate(
    front_model: CandidateRanker,
    back_model: CandidateRanker,
    samples: Sequence[Sample],
    stats: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    front_model.eval()
    back_model.eval()
    front_x, front_y, back_x, back_y = stack_samples(samples)
    loss = batch_loss(
        front_model,
        back_model,
        (front_x, front_y, back_x, back_y),
        stats,
    )
    front_logits = front_model(normalize_features(front_x, stats["front"]))
    back_logits = back_model(normalize_features(back_x, stats["back"]))
    front_top = front_logits.topk(FRONT_PICK, dim=1).indices
    back_top = back_logits.topk(BACK_PICK, dim=1).indices
    front_hits = front_y.gather(1, front_top).sum(dim=1)
    back_hits = back_y.gather(1, back_top).sum(dim=1)
    return {
        "loss": float(loss),
        "front_hits": float(front_hits.mean()),
        "back_hits": float(back_hits.mean()),
        "total_hits": float((front_hits + back_hits).mean()),
    }


def train_stage(
    front_model: CandidateRanker,
    back_model: CandidateRanker,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    stats: dict[str, dict[str, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
    stage_name: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], int, dict[str, float]]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        list(front_model.parameters()) + list(back_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_tensors = stack_samples(train_samples)
    best_states = model_states(front_model, back_model)
    best_epoch = 0
    best_metrics = evaluate(front_model, back_model, validation_samples, stats)
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        front_model.train()
        back_model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_samples), generator=generator)

        for start in range(0, len(train_samples), batch_size):
            indices = order[start : start + batch_size]
            batch = tuple(values[indices] for values in train_tensors)
            optimizer.zero_grad()
            loss = batch_loss(front_model, back_model, batch, stats)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(front_model.parameters()) + list(back_model.parameters()), 1.0
            )
            optimizer.step()

        metrics = evaluate(front_model, back_model, validation_samples, stats)
        if metrics["loss"] < best_metrics["loss"] - 1e-6:
            best_metrics = metrics
            best_states = model_states(front_model, back_model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    restore_states(best_states, front_model, back_model)
    print(
        f"{stage_name}: best_epoch={best_epoch}, "
        f"val_loss={best_metrics['loss']:.4f}, "
        f"hits={best_metrics['front_hits']:.3f}+{best_metrics['back_hits']:.3f}"
    )
    return best_states, best_epoch, best_metrics


def refine_for_deployment(
    front_model: CandidateRanker,
    back_model: CandidateRanker,
    samples: Sequence[Sample],
    stats: dict[str, dict[str, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> None:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        list(front_model.parameters()) + list(back_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    tensors = stack_samples(samples)

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(samples), generator=generator)
        for start in range(0, len(samples), batch_size):
            indices = order[start : start + batch_size]
            batch = tuple(values[indices] for values in tensors)
            optimizer.zero_grad()
            loss = batch_loss(front_model, back_model, batch, stats)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(front_model.parameters()) + list(back_model.parameters()), 1.0
            )
            optimizer.step()


def frequency_baseline(
    draws: Sequence[Draw], validation_dates: set[str], window: int = 50
) -> dict[str, float]:
    front_hits: list[int] = []
    back_hits: list[int] = []
    for index, target in enumerate(draws):
        if target.date not in validation_dates:
            continue
        history = draws[max(0, index - window) : index]
        front_counts = [0] * FRONT_SIZE
        back_counts = [0] * BACK_SIZE
        for draw in history:
            for value in draw.front:
                front_counts[value - 1] += 1
            for value in draw.back:
                back_counts[value - 1] += 1
        predicted_front = {
            index + 1
            for index in sorted(
                range(FRONT_SIZE), key=lambda item: (-front_counts[item], item)
            )[:FRONT_PICK]
        }
        predicted_back = {
            index + 1
            for index in sorted(
                range(BACK_SIZE), key=lambda item: (-back_counts[item], item)
            )[:BACK_PICK]
        }
        front_hits.append(len(predicted_front.intersection(target.front)))
        back_hits.append(len(predicted_back.intersection(target.back)))

    return {
        "front_hits": sum(front_hits) / len(front_hits),
        "back_hits": sum(back_hits) / len(back_hits),
        "total_hits": (sum(front_hits) + sum(back_hits)) / len(front_hits),
    }


def data_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(
    data_path: Path,
    output_path: Path,
    *,
    validation_draws: int,
    test_draws: int,
    min_history: int,
    hidden_dim: int,
    seed: int,
) -> dict:
    set_seed(seed)
    draws = load_draws(data_path)
    samples = build_samples(draws, min_history=min_history)
    minimum_training_samples = 200
    if len(samples) < minimum_training_samples + validation_draws + test_draws:
        raise ValueError(
            "Not enough samples for chronological train/validation/test split"
        )

    test = samples[-test_draws:]
    validation = samples[-(validation_draws + test_draws) : -test_draws]
    train = samples[: -(validation_draws + test_draws)]
    stats = normalization(train)
    features = feature_names()
    front_model = CandidateRanker(len(features), hidden_dim=hidden_dim)
    back_model = CandidateRanker(len(features), hidden_dim=hidden_dim)

    evaluation_states, best_epoch, validation_metrics = train_stage(
        front_model,
        back_model,
        train,
        validation,
        stats,
        epochs=250,
        learning_rate=5e-3,
        weight_decay=1e-3,
        batch_size=128,
        patience=30,
        seed=seed,
        stage_name="model_selection",
    )
    test_metrics = evaluate(front_model, back_model, test, stats)

    baseline = frequency_baseline(
        draws, {sample.date for sample in test}, window=50
    )
    random_expected = {
        "front_hits": FRONT_PICK * FRONT_PICK / FRONT_SIZE,
        "back_hits": BACK_PICK * BACK_PICK / BACK_SIZE,
    }
    random_expected["total_hits"] = (
        random_expected["front_hits"] + random_expected["back_hits"]
    )

    deployment_epochs = max(1, min(best_epoch, 30))
    refine_for_deployment(
        front_model,
        back_model,
        samples,
        stats,
        epochs=deployment_epochs,
        learning_rate=5e-4,
        weight_decay=1e-3,
        batch_size=64,
        seed=seed + 10_000,
    )

    checkpoint = {
        "format_version": FORMAT_VERSION,
        "model_type": "shared_candidate_ranker",
        "config": {
            "front_size": FRONT_SIZE,
            "back_size": BACK_SIZE,
            "front_pick": FRONT_PICK,
            "back_pick": BACK_PICK,
            "windows": list(DEFAULT_WINDOWS),
            "half_lives": list(DEFAULT_HALF_LIVES),
            "feature_names": features,
            "hidden_dim": hidden_dim,
            "min_history": min_history,
        },
        "front_state": front_model.state_dict(),
        "back_state": back_model.state_dict(),
        "evaluation_state": {
            "front": evaluation_states["front"],
            "back": evaluation_states["back"],
        },
        "normalization": stats,
        "metadata": {
            "seed": seed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(data_path),
            "data_sha256": data_sha256(data_path),
            "draw_count": len(draws),
            "sample_count": len(samples),
            "training_cutoff": draws[-1].date,
            "latest_issue": draws[-1].issue,
            "training_samples": len(train),
            "validation_draws": validation_draws,
            "test_draws": test_draws,
            "best_epoch": best_epoch,
            "deployment_refine_epochs": deployment_epochs,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "frequency_50_baseline": baseline,
            "random_expected": random_expected,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def train_member_job(job: dict) -> str:
    torch.set_num_threads(1)
    output_path = Path(job["output_path"])
    train_model(
        Path(job["data_path"]),
        output_path,
        validation_draws=job["validation_draws"],
        test_draws=job["test_draws"],
        min_history=job["min_history"],
        hidden_dim=job["hidden_dim"],
        seed=job["seed"],
    )
    return str(output_path)


def instantiate_models(
    config: dict,
    front_state: dict[str, torch.Tensor],
    back_state: dict[str, torch.Tensor],
) -> tuple[CandidateRanker, CandidateRanker]:
    feature_count = len(config["feature_names"])
    front_model = CandidateRanker(feature_count, hidden_dim=config["hidden_dim"])
    back_model = CandidateRanker(feature_count, hidden_dim=config["hidden_dim"])
    front_model.load_state_dict(front_state)
    back_model.load_state_dict(back_state)
    front_model.eval()
    back_model.eval()
    return front_model, back_model


@torch.no_grad()
def metrics_from_logits(
    front_logits: torch.Tensor,
    back_logits: torch.Tensor,
    front_targets: torch.Tensor,
    back_targets: torch.Tensor,
) -> dict[str, float]:
    loss = nn.functional.binary_cross_entropy_with_logits(
        front_logits, front_targets
    ) + nn.functional.binary_cross_entropy_with_logits(back_logits, back_targets)
    front_top = front_logits.topk(FRONT_PICK, dim=1).indices
    back_top = back_logits.topk(BACK_PICK, dim=1).indices
    front_hits = front_targets.gather(1, front_top).sum(dim=1)
    back_hits = back_targets.gather(1, back_top).sum(dim=1)
    return {
        "loss": float(loss),
        "front_hits": float(front_hits.mean()),
        "back_hits": float(back_hits.mean()),
        "total_hits": float((front_hits + back_hits).mean()),
    }


def build_ensemble(
    data_path: Path, member_paths: Sequence[Path], output_path: Path
) -> dict:
    if len(member_paths) < 2:
        raise ValueError("An ensemble requires at least two member checkpoints")
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in member_paths
    ]
    first = checkpoints[0]
    config = first["config"]
    metadata = first["metadata"]
    stats = first["normalization"]

    for path, checkpoint in zip(member_paths, checkpoints):
        if checkpoint.get("model_type") != "shared_candidate_ranker":
            raise ValueError(f"Not a single-model checkpoint: {path}")
        if "evaluation_state" not in checkpoint:
            raise ValueError(f"Checkpoint lacks evaluation state: {path}")
        if checkpoint["config"] != config:
            raise ValueError(f"Incompatible model config: {path}")
        if checkpoint["metadata"]["data_sha256"] != metadata["data_sha256"]:
            raise ValueError(f"Checkpoint was trained on different data: {path}")
        if checkpoint["metadata"]["training_cutoff"] != metadata["training_cutoff"]:
            raise ValueError(f"Checkpoint has a different cutoff: {path}")
        for zone in ("front", "back"):
            for statistic in ("mean", "std"):
                if not torch.allclose(
                    checkpoint["normalization"][zone][statistic],
                    stats[zone][statistic],
                ):
                    raise ValueError(f"Normalization differs for {path}")

    draws = load_draws(data_path)
    samples = build_samples(draws, min_history=config["min_history"])
    test_draws = metadata["test_draws"]
    test = samples[-test_draws:]
    front_x, front_y, back_x, back_y = stack_samples(test)
    front_logits_members: list[torch.Tensor] = []
    back_logits_members: list[torch.Tensor] = []

    for checkpoint in checkpoints:
        evaluation_state = checkpoint["evaluation_state"]
        front_model, back_model = instantiate_models(
            config, evaluation_state["front"], evaluation_state["back"]
        )
        front_logits_members.append(
            front_model(normalize_features(front_x, stats["front"]))
        )
        back_logits_members.append(
            back_model(normalize_features(back_x, stats["back"]))
        )

    test_metrics = metrics_from_logits(
        torch.stack(front_logits_members).mean(dim=0),
        torch.stack(back_logits_members).mean(dim=0),
        front_y,
        back_y,
    )
    ensemble = {
        "format_version": FORMAT_VERSION,
        "model_type": "shared_candidate_ranker_ensemble",
        "config": config,
        "members": [
            {
                "seed": checkpoint["metadata"]["seed"],
                "front_state": checkpoint["front_state"],
                "back_state": checkpoint["back_state"],
            }
            for checkpoint in checkpoints
        ],
        "normalization": stats,
        "metadata": {
            **metadata,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "member_count": len(checkpoints),
            "member_seeds": [
                checkpoint["metadata"]["seed"] for checkpoint in checkpoints
            ],
            "member_validation_metrics": [
                checkpoint["metadata"]["validation_metrics"]
                for checkpoint in checkpoints
            ],
            "member_test_metrics": [
                checkpoint["metadata"]["test_metrics"]
                for checkpoint in checkpoints
            ],
            "test_metrics": test_metrics,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ensemble, output_path)
    return ensemble


def load_models(
    path: Path,
) -> tuple[list[tuple[CandidateRanker, CandidateRanker]], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    config = checkpoint["config"]
    if checkpoint["model_type"] == "shared_candidate_ranker_ensemble":
        states = [
            (member["front_state"], member["back_state"])
            for member in checkpoint["members"]
        ]
    elif checkpoint["model_type"] == "shared_candidate_ranker":
        states = [(checkpoint["front_state"], checkpoint["back_state"])]
    else:
        raise ValueError(f"Unsupported model type: {checkpoint['model_type']}")
    return [
        instantiate_models(config, front_state, back_state)
        for front_state, back_state in states
    ], checkpoint


def load_model(
    path: Path,
) -> tuple[CandidateRanker, CandidateRanker, dict]:
    models, checkpoint = load_models(path)
    if len(models) != 1:
        raise ValueError("Checkpoint contains an ensemble; use load_models")
    front_model, back_model = models[0]
    return front_model, back_model, checkpoint


@torch.no_grad()
def predict_next(data_path: Path, model_path: Path) -> dict:
    draws = load_draws(data_path)
    models, checkpoint = load_models(model_path)
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    current_hash = data_sha256(data_path)
    if current_hash != metadata["data_sha256"]:
        raise ValueError(
            "Model checkpoint does not match the current dataset; retrain before predicting"
        )
    if draws[-1].date != metadata["training_cutoff"]:
        raise ValueError(
            "Model checkpoint is stale for the latest draw; retrain before predicting"
        )
    latest_segment = split_contiguous_segments(draws)[-1]
    if len(latest_segment) < config["min_history"]:
        raise ValueError("Not enough contiguous recent history for prediction")

    front_history = [draw.front for draw in latest_segment]
    back_history = [draw.back for draw in latest_segment]
    front_x = zone_features(
        front_history,
        config["front_size"],
        config["windows"],
        config["half_lives"],
    )
    back_x = zone_features(
        back_history,
        config["back_size"],
        config["windows"],
        config["half_lives"],
    )
    stats = checkpoint["normalization"]
    normalized_front = normalize_features(front_x, stats["front"])
    normalized_back = normalize_features(back_x, stats["back"])
    front_logits = torch.stack(
        [front_model(normalized_front) for front_model, _ in models]
    ).mean(dim=0)
    back_logits = torch.stack(
        [back_model(normalized_back) for _, back_model in models]
    ).mean(dim=0)
    front_probabilities = torch.sigmoid(front_logits)
    back_probabilities = torch.sigmoid(back_logits)
    front_indices = front_logits.topk(config["front_pick"]).indices.tolist()
    back_indices = back_logits.topk(config["back_pick"]).indices.tolist()

    return {
        "front": sorted(index + 1 for index in front_indices),
        "back": sorted(index + 1 for index in back_indices),
        "front_probabilities": front_probabilities.tolist(),
        "back_probabilities": back_probabilities.tolist(),
        "latest_issue": draws[-1].issue,
        "latest_date": draws[-1].date,
        "model_cutoff": metadata["training_cutoff"],
        "method": (
            "model_v2_ensemble"
            if checkpoint["model_type"] == "shared_candidate_ranker_ensemble"
            else "model_v2"
        ),
    }


def print_prediction(prediction: dict) -> None:
    front = " ".join(f"{value:02d}" for value in prediction["front"])
    back = " ".join(f"{value:02d}" for value in prediction["back"])
    print(
        f"Latest input: {prediction['latest_issue']} ({prediction['latest_date']})"
    )
    print(f"Model cutoff: {prediction['model_cutoff']}")
    print(f"Prediction: {front} + {back} ({prediction['method']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train a fresh v2 model")
    parser.add_argument(
        "--train-ensemble",
        type=int,
        metavar="MEMBERS",
        help="Train a fixed-seed ensemble in parallel and save it as --model",
    )
    parser.add_argument("--predict", action="store_true", help="Run model inference")
    parser.add_argument(
        "--ensemble-members",
        type=Path,
        nargs="+",
        help="Build --model by averaging these fixed member checkpoints",
    )
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--validation-draws", type=int, default=36)
    parser.add_argument("--test-draws", type=int, default=36)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_modes = sum(
        value is not None and value is not False
        for value in (args.train, args.train_ensemble, args.ensemble_members)
    )
    if training_modes > 1:
        raise ValueError(
            "--train, --train-ensemble and --ensemble-members are mutually exclusive"
        )
    if (
        not args.train
        and not args.train_ensemble
        and not args.predict
        and not args.ensemble_members
    ):
        args.predict = True

    if args.train_ensemble:
        if args.train_ensemble < 2:
            raise ValueError("--train-ensemble requires at least two members")
        runs_dir = args.model.parent / "runs"
        jobs = []
        for offset in range(args.train_ensemble):
            seed = args.seed + offset
            member_path = runs_dir / f"{args.model.stem}_seed_{seed}{args.model.suffix}"
            jobs.append(
                {
                    "data_path": str(args.data),
                    "output_path": str(member_path),
                    "validation_draws": args.validation_draws,
                    "test_draws": args.test_draws,
                    "min_history": args.min_history,
                    "hidden_dim": args.hidden_dim,
                    "seed": seed,
                }
            )
        worker_count = min(
            args.train_ensemble, max(1, (os.cpu_count() or 2) // 2)
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            member_paths = [Path(path) for path in executor.map(train_member_job, jobs)]
        checkpoint = build_ensemble(args.data, member_paths, args.model)
        metadata = checkpoint["metadata"]
        print(
            f"Saved {args.model} ({metadata['member_count']} members, "
            f"holdout_hits={metadata['test_metrics']['total_hits']:.3f})"
        )

    if args.ensemble_members:
        checkpoint = build_ensemble(args.data, args.ensemble_members, args.model)
        metadata = checkpoint["metadata"]
        print(
            f"Saved {args.model} ({metadata['member_count']} members, "
            f"holdout_hits={metadata['test_metrics']['total_hits']:.3f})"
        )

    if args.train:
        checkpoint = train_model(
            args.data,
            args.model,
            validation_draws=args.validation_draws,
            test_draws=args.test_draws,
            min_history=args.min_history,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        metadata = checkpoint["metadata"]
        print(
            f"Saved {args.model} ({sum(value.numel() for value in checkpoint['front_state'].values()) + sum(value.numel() for value in checkpoint['back_state'].values())} parameters)"
        )
        print(
            "Untouched temporal holdout: "
            f"model={metadata['test_metrics']['total_hits']:.3f}, "
            f"frequency50={metadata['frequency_50_baseline']['total_hits']:.3f}, "
            f"random_expected={metadata['random_expected']['total_hits']:.3f}"
        )

    if args.predict or args.train_ensemble or args.ensemble_members:
        print_prediction(predict_next(args.data, args.model))


if __name__ == "__main__":
    main()
