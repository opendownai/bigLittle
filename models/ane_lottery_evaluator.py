#!/usr/bin/env python3
"""Evaluate an ANE dynamic-training checkpoint on chronological held-out draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "dlt_merged.json"
DEFAULT_TRAINING_MANIFEST = (
    PROJECT_ROOT / "data" / "lottery_train.bin.manifest.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "ane_training"
    / "upstream"
    / "training"
    / "training_dynamic"
    / "ane_lottery_dyn_ckpt.bin"
)
DEFAULT_DEPLOYMENT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "ane_lottery_deploy.bin"
)
DEFAULT_DEPLOYMENT_MANIFEST = (
    PROJECT_ROOT / "models" / "ane_lottery_deploy.manifest.json"
)
HEADER = struct.Struct("<10i2f3d6i")
MAGIC = 0x424C5A54
CHECKPOINT_VERSION = 4
FRONT_SIZE = 35
BACK_SIZE = 12
BACK_TOKEN_OFFSET = 34
SEPARATOR_TOKEN = 47


@dataclass(frozen=True)
class CheckpointConfig:
    step: int
    total_steps: int
    layers: int
    vocab: int
    dim: int
    hidden: int
    heads: int
    sequence: int
    learning_rate: float
    training_loss: float
    kv_heads: int
    head_dim: int
    q_dim: int


def read_array(handle: Any, count: int) -> torch.Tensor:
    values = np.fromfile(handle, dtype="<f4", count=count)
    if values.size != count:
        raise ValueError("Checkpoint ended before all weights were read")
    return torch.from_numpy(values.copy())


def skip_arrays(handle: Any, count: int) -> None:
    handle.seek(count * 4, 1)


def load_checkpoint(
    path: Path,
) -> tuple[CheckpointConfig, list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    with path.open("rb") as handle:
        raw_header = handle.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("Checkpoint header is truncated")
        values = HEADER.unpack(raw_header)
        (
            magic,
            version,
            step,
            total_steps,
            layers,
            vocab,
            dim,
            hidden,
            heads,
            sequence,
            learning_rate,
            training_loss,
            _cum_compile,
            _cum_train,
            _cum_wall,
            _cum_steps,
            _cum_batches,
            _adam_t,
            kv_heads,
            head_dim,
            q_dim,
        ) = values
        if magic != MAGIC or version != CHECKPOINT_VERSION:
            raise ValueError("Unsupported ANE checkpoint")
        if q_dim != heads * head_dim or dim % heads or heads % kv_heads:
            raise ValueError("Inconsistent checkpoint dimensions")

        config = CheckpointConfig(
            step=step,
            total_steps=total_steps,
            layers=layers,
            vocab=vocab,
            dim=dim,
            hidden=hidden,
            heads=heads,
            sequence=sequence,
            learning_rate=learning_rate,
            training_loss=training_loss,
            kv_heads=kv_heads,
            head_dim=head_dim,
            q_dim=q_dim,
        )

        weight_counts = (
            q_dim * dim,
            kv_heads * head_dim * dim,
            kv_heads * head_dim * dim,
            dim * q_dim,
            hidden * dim,
            dim * hidden,
            hidden * dim,
            dim,
            dim,
        )
        names = (
            "wq",
            "wk",
            "wv",
            "wo",
            "w1",
            "w2",
            "w3",
            "rms_attention",
            "rms_ffn",
        )
        layer_weights: list[dict[str, torch.Tensor]] = []
        for _ in range(layers):
            weights = {
                name: read_array(handle, count)
                for name, count in zip(names, weight_counts)
            }
            for count in weight_counts:
                skip_arrays(handle, count * 2)
            weights["wq"] = weights["wq"].reshape(q_dim, dim)
            weights["wk"] = weights["wk"].reshape(
                kv_heads * head_dim, dim
            )
            weights["wv"] = weights["wv"].reshape(
                kv_heads * head_dim, dim
            )
            weights["wo"] = weights["wo"].reshape(dim, q_dim)
            weights["w1"] = weights["w1"].reshape(hidden, dim)
            weights["w2"] = weights["w2"].reshape(dim, hidden)
            weights["w3"] = weights["w3"].reshape(hidden, dim)
            layer_weights.append(weights)

        rms_final = read_array(handle, dim)
        skip_arrays(handle, dim * 2)
        embedding = read_array(handle, vocab * dim).reshape(vocab, dim)
        skip_arrays(handle, vocab * dim * 2)
        if handle.read(1):
            raise ValueError("Checkpoint contains unexpected trailing bytes")

    all_weights = [rms_final, embedding]
    for layer in layer_weights:
        all_weights.extend(layer.values())
    if not all(torch.isfinite(weight).all() for weight in all_weights):
        raise ValueError("Checkpoint contains a non-finite weight")
    return config, layer_weights, rms_final, embedding


class AneLotteryModel:
    def __init__(
        self,
        config: CheckpointConfig,
        layers: list[dict[str, torch.Tensor]],
        rms_final: torch.Tensor,
        embedding: torch.Tensor,
    ):
        self.config = config
        self.layers = layers
        self.rms_final = rms_final
        self.embedding = embedding
        self.residual_scale = 1.0 / math.sqrt(2.0 * config.layers)

    @staticmethod
    def rms_norm(
        values: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
        return values * scale * weights

    def rope(self, values: torch.Tensor) -> torch.Tensor:
        sequence = values.shape[1]
        positions = torch.arange(sequence, dtype=values.dtype)
        pair_indices = torch.arange(
            self.config.head_dim // 2, dtype=values.dtype
        )
        frequencies = 1.0 / (
            10000.0 ** (2.0 * pair_indices / self.config.head_dim)
        )
        angles = positions[:, None] * frequencies[None, :]
        cosine = torch.cos(angles)[None, :, :]
        sine = torch.sin(angles)[None, :, :]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated = torch.empty_like(values)
        rotated[..., 0::2] = even * cosine - odd * sine
        rotated[..., 1::2] = odd * cosine + even * sine
        return rotated

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 1 or not 1 <= len(tokens) <= self.config.sequence:
            raise ValueError("Token context length is outside checkpoint limits")
        values = self.embedding[tokens]
        for layer in self.layers:
            normalized = self.rms_norm(
                values, layer["rms_attention"]
            )
            query = F.linear(normalized, layer["wq"])
            key = F.linear(normalized, layer["wk"])
            value = F.linear(normalized, layer["wv"])

            query = query.reshape(
                len(tokens), self.config.heads, self.config.head_dim
            ).permute(1, 0, 2)
            key = key.reshape(
                len(tokens), self.config.kv_heads, self.config.head_dim
            ).permute(1, 0, 2)
            value = value.reshape(
                len(tokens), self.config.kv_heads, self.config.head_dim
            ).permute(1, 0, 2)
            query = self.rope(query)
            key = self.rope(key)

            ratio = self.config.heads // self.config.kv_heads
            if ratio > 1:
                key = key.repeat(ratio, 1, 1)
                value = value.repeat(ratio, 1, 1)

            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(self.config.head_dim)
            causal_mask = torch.triu(
                torch.ones(
                    len(tokens), len(tokens), dtype=torch.bool
                ),
                diagonal=1,
            )
            scores.masked_fill_(causal_mask[None, :, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            attended = torch.matmul(attention, value)
            attended = attended.permute(1, 0, 2).reshape(
                len(tokens), self.config.q_dim
            )
            attention_output = F.linear(attended, layer["wo"])
            residual = values + self.residual_scale * attention_output

            normalized = self.rms_norm(residual, layer["rms_ffn"])
            hidden_1 = F.linear(normalized, layer["w1"])
            hidden_3 = F.linear(normalized, layer["w3"])
            feed_forward = F.linear(
                F.silu(hidden_1) * hidden_3, layer["w2"]
            )
            values = residual + self.residual_scale * feed_forward

        normalized = self.rms_norm(values, self.rms_final)
        return F.linear(normalized, self.embedding)

    def next_logits(self, context: list[int]) -> torch.Tensor:
        limited = context[-self.config.sequence :]
        tokens = torch.tensor(limited, dtype=torch.long)
        return self.forward(tokens)[-1]


def draw_tokens(draw: dict[str, Any]) -> list[int]:
    front = sorted(int(value) for value in draw["frontBalls"])
    back = sorted(int(value) for value in draw["backBalls"])
    return (
        [value - 1 for value in front]
        + [BACK_TOKEN_OFFSET + value for value in back]
        + [SEPARATOR_TOKEN]
    )


def valid_front_tokens(prefix: list[int], position: int) -> list[int]:
    start = prefix[-1] + 1 if prefix else 0
    remaining = 4 - position
    stop_inclusive = FRONT_SIZE - 1 - remaining
    return list(range(start, stop_inclusive + 1))


def valid_back_tokens(prefix: list[int], position: int) -> list[int]:
    start = prefix[-1] + 1 if prefix else BACK_TOKEN_OFFSET + 1
    remaining = 1 - position
    stop_inclusive = BACK_TOKEN_OFFSET + BACK_SIZE - remaining
    return list(range(start, stop_inclusive + 1))


def masked_loss(
    model: AneLotteryModel,
    historical_context: list[int],
    actual_tokens: list[int],
) -> float:
    prefix: list[int] = []
    losses: list[float] = []
    for position, actual in enumerate(actual_tokens[:5]):
        candidates = valid_front_tokens(prefix, position)
        logits = model.next_logits(historical_context + prefix)
        candidate_logits = logits[candidates]
        target_index = candidates.index(actual)
        losses.append(
            float(-F.log_softmax(candidate_logits, dim=0)[target_index])
        )
        prefix.append(actual)

    back_prefix: list[int] = []
    for position, actual in enumerate(actual_tokens[5:7]):
        candidates = valid_back_tokens(back_prefix, position)
        logits = model.next_logits(
            historical_context + prefix + back_prefix
        )
        candidate_logits = logits[candidates]
        target_index = candidates.index(actual)
        losses.append(
            float(-F.log_softmax(candidate_logits, dim=0)[target_index])
        )
        back_prefix.append(actual)
    return sum(losses) / len(losses)


def predict_draw(
    model: AneLotteryModel, historical_context: list[int]
) -> tuple[list[int], list[int]]:
    front_tokens: list[int] = []
    for position in range(5):
        candidates = valid_front_tokens(front_tokens, position)
        logits = model.next_logits(historical_context + front_tokens)
        best = max(candidates, key=lambda token: float(logits[token]))
        front_tokens.append(best)

    back_tokens: list[int] = []
    for position in range(2):
        candidates = valid_back_tokens(back_tokens, position)
        logits = model.next_logits(
            historical_context + front_tokens + back_tokens
        )
        best = max(candidates, key=lambda token: float(logits[token]))
        back_tokens.append(best)

    front = [token + 1 for token in front_tokens]
    back = [token - BACK_TOKEN_OFFSET for token in back_tokens]
    return front, back


def frequency_prediction(
    draws: list[dict[str, Any]], target_index: int, window: int = 50
) -> tuple[list[int], list[int]]:
    front_counts = [0] * FRONT_SIZE
    back_counts = [0] * BACK_SIZE
    for draw in draws[max(0, target_index - window) : target_index]:
        for value in draw["frontBalls"]:
            front_counts[int(value) - 1] += 1
        for value in draw["backBalls"]:
            back_counts[int(value) - 1] += 1
    front = sorted(
        sorted(
            range(1, FRONT_SIZE + 1),
            key=lambda value: (-front_counts[value - 1], value),
        )[:5]
    )
    back = sorted(
        sorted(
            range(1, BACK_SIZE + 1),
            key=lambda value: (-back_counts[value - 1], value),
        )[:2]
    )
    return front, back


def evaluate_range(
    model: AneLotteryModel,
    draws: list[dict[str, Any]],
    start: int,
    stop: int,
) -> dict[str, Any]:
    all_tokens = [
        token for draw in draws for token in draw_tokens(draw)
    ]
    details: list[dict[str, Any]] = []
    model_front_hits = 0
    model_back_hits = 0
    frequency_front_hits = 0
    frequency_back_hits = 0
    losses: list[float] = []

    for target_index in range(start, stop):
        draw = draws[target_index]
        context = all_tokens[: target_index * 8]
        actual_tokens = draw_tokens(draw)
        predicted_front, predicted_back = predict_draw(model, context)
        frequency_front, frequency_back = frequency_prediction(
            draws, target_index
        )
        actual_front = {int(value) for value in draw["frontBalls"]}
        actual_back = {int(value) for value in draw["backBalls"]}
        front_hits = len(actual_front.intersection(predicted_front))
        back_hits = len(actual_back.intersection(predicted_back))
        frequency_front_hit = len(
            actual_front.intersection(frequency_front)
        )
        frequency_back_hit = len(actual_back.intersection(frequency_back))
        loss = masked_loss(model, context, actual_tokens)

        model_front_hits += front_hits
        model_back_hits += back_hits
        frequency_front_hits += frequency_front_hit
        frequency_back_hits += frequency_back_hit
        losses.append(loss)
        details.append(
            {
                "issue": str(draw["issueNumber"]),
                "date": str(draw["date"]),
                "prediction": {
                    "front": predicted_front,
                    "back": predicted_back,
                },
                "frontHits": front_hits,
                "backHits": back_hits,
                "maskedNumberNll": loss,
            }
        )

    count = stop - start
    return {
        "draws": count,
        "fromIssue": details[0]["issue"],
        "toIssue": details[-1]["issue"],
        "model": {
            "frontHits": model_front_hits / count,
            "backHits": model_back_hits / count,
            "totalHits": (model_front_hits + model_back_hits) / count,
            "maskedNumberNll": sum(losses) / len(losses),
        },
        "frequency50": {
            "frontHits": frequency_front_hits / count,
            "backHits": frequency_back_hits / count,
            "totalHits": (
                frequency_front_hits + frequency_back_hits
            )
            / count,
        },
        "randomExpected": {
            "frontHits": 25 / 35,
            "backHits": 4 / 12,
            "totalHits": 25 / 35 + 4 / 12,
        },
        "details": details,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_deployment_identity(
    deployment_manifest_path: Path,
    data_path: Path,
    checkpoint_path: Path,
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    deployment = json.loads(
        deployment_manifest_path.read_text(encoding="utf-8")
    )
    expected = {
        "sourceDataSha256": file_sha256(data_path),
        "trainingDataSha256": str(training_manifest["data_sha256"]),
        "checkpointSha256": file_sha256(checkpoint_path),
        "sourceDrawCount": int(training_manifest["source_draw_count"]),
        "trainingDrawCount": int(training_manifest["draw_count"]),
        "latestIssue": str(training_manifest["latest_issue"]),
        "latestDate": str(training_manifest["date_to"]),
    }
    differences = {
        key: {"manifest": deployment.get(key), "actual": value}
        for key, value in expected.items()
        if deployment.get(key) != value
    }
    if differences:
        raise ValueError(
            "Deployment manifest does not match current artifacts: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    return deployment


def validate_deployment_config(
    deployment: dict[str, Any], config: CheckpointConfig
) -> None:
    actual = {
        "step": config.step,
        "layers": config.layers,
        "dim": config.dim,
        "hidden": config.hidden,
        "heads": config.heads,
        "sequence": config.sequence,
        "vocab": config.vocab,
    }
    expected = deployment.get("config")
    if expected != actual:
        raise ValueError(
            "Deployment manifest config does not match checkpoint: "
            f"manifest={expected!r}, actual={actual!r}"
        )


def predict_next_from_files(
    data_path: Path = DEFAULT_DATA,
    checkpoint_path: Path = DEFAULT_DEPLOYMENT_CHECKPOINT,
    manifest_path: Path = DEFAULT_TRAINING_MANIFEST,
    deployment_manifest_path: Path = DEFAULT_DEPLOYMENT_MANIFEST,
) -> dict[str, Any]:
    """Predict after the latest audited draw with a full-data checkpoint."""
    draws = json.loads(data_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != file_sha256(data_path):
        raise ValueError("Training manifest does not match the current dataset")
    if int(manifest["source_draw_count"]) != len(draws):
        raise ValueError("Training manifest source count is stale")
    if int(manifest["holdout_draw_count"]) != 0:
        raise ValueError("Prediction checkpoint was not trained on all draws")
    if int(manifest["draw_count"]) != len(draws):
        raise ValueError("Prediction checkpoint has a partial training window")
    if draws[-1]["date"] != manifest["date_to"]:
        raise ValueError("Training cutoff does not match the latest draw")

    deployment = validate_deployment_identity(
        deployment_manifest_path,
        data_path,
        checkpoint_path,
        manifest,
    )
    torch.set_num_threads(1)
    config, layers, rms_final, embedding = load_checkpoint(checkpoint_path)
    validate_deployment_config(deployment, config)
    model = AneLotteryModel(config, layers, rms_final, embedding)
    historical_context = [
        token for draw in draws for token in draw_tokens(draw)
    ]
    front, back = predict_draw(model, historical_context)
    return {
        "latestInputIssue": str(draws[-1]["issueNumber"]),
        "latestInputDate": str(draws[-1]["date"]),
        "front": front,
        "back": back,
        "method": "ane_autoregressive_masked",
        "checkpointSha256": file_sha256(checkpoint_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=DEFAULT_TRAINING_MANIFEST,
    )
    parser.add_argument(
        "--deployment-manifest",
        type=Path,
        default=DEFAULT_DEPLOYMENT_MANIFEST,
    )
    parser.add_argument("--validation-draws", type=int, default=36)
    parser.add_argument("--test-draws", type=int, default=36)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "both"),
        default="both",
    )
    parser.add_argument(
        "--predict-next",
        action="store_true",
        help="predict after the latest draw using a checkpoint trained on all data",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    draws = json.loads(args.data.read_text(encoding="utf-8"))
    manifest = json.loads(
        args.training_manifest.read_text(encoding="utf-8")
    )
    if manifest["source_sha256"] != file_sha256(args.data):
        raise ValueError("Training manifest does not match the current dataset")
    if manifest["source_draw_count"] != len(draws):
        raise ValueError("Training manifest source count is stale")
    if args.predict_next:
        if manifest["holdout_draw_count"] != 0:
            raise ValueError("Prediction checkpoint was not trained on all draws")
    elif (
        args.validation_draws + args.test_draws
        != manifest["holdout_draw_count"]
    ):
        raise ValueError("Evaluation split does not match the ANE holdout")

    training_draws = int(manifest["draw_count"])
    if draws[training_draws - 1]["date"] != manifest["date_to"]:
        raise ValueError("Training cutoff does not match the dataset")

    deployment: dict[str, Any] | None = None
    if args.predict_next:
        deployment = validate_deployment_identity(
            args.deployment_manifest,
            args.data,
            args.checkpoint,
            manifest,
        )

    config, layers, rms_final, embedding = load_checkpoint(args.checkpoint)
    if deployment is not None:
        validate_deployment_config(deployment, config)
    model = AneLotteryModel(config, layers, rms_final, embedding)
    report: dict[str, Any] = {
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "step": config.step,
            "trainingLoss": config.training_loss,
            "layers": config.layers,
            "dim": config.dim,
            "hidden": config.hidden,
            "heads": config.heads,
            "parameters": sum(
                weight.numel()
                for layer in layers
                for weight in layer.values()
            )
            + rms_final.numel()
            + embedding.numel(),
        },
        "data": {
            "records": len(draws),
            "trainingRecords": training_draws,
            "trainingCutoffIssue": draws[training_draws - 1]["issueNumber"],
            "trainingCutoffDate": draws[training_draws - 1]["date"],
        },
    }

    if args.predict_next:
        historical_context = [
            token for draw in draws for token in draw_tokens(draw)
        ]
        front, back = predict_draw(model, historical_context)
        report["nextPrediction"] = {
            "latestInputIssue": str(draws[-1]["issueNumber"]),
            "latestInputDate": str(draws[-1]["date"]),
            "front": front,
            "back": back,
            "method": "ane_autoregressive_masked",
        }
    else:
        validation_start = training_draws
        validation_stop = validation_start + args.validation_draws
        test_start = validation_stop
        test_stop = test_start + args.test_draws
        if args.split in ("validation", "both"):
            report["validation"] = evaluate_range(
                model, draws, validation_start, validation_stop
            )
        if args.split in ("test", "both"):
            report["test"] = evaluate_range(
                model, draws, test_start, test_stop
            )

    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Wrote {args.output}")

    summary = {
        "checkpoint": report["checkpoint"],
        **{
            split: {
                "draws": report[split]["draws"],
                "fromIssue": report[split]["fromIssue"],
                "toIssue": report[split]["toIssue"],
                "model": report[split]["model"],
                "frequency50": report[split]["frequency50"],
                "randomExpected": report[split]["randomExpected"],
            }
            for split in ("validation", "test")
            if split in report
        },
    }
    if "nextPrediction" in report:
        summary["nextPrediction"] = report["nextPrediction"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
