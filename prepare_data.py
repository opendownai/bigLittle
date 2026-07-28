#!/usr/bin/env python3
"""
Prepare lottery data for training
Converts CSV lottery history to token sequence
"""

import json
import argparse
import hashlib
import struct
from pathlib import Path

FRONT_BALLS = 35
BACK_BALLS = 12
BACK_TOKEN_OFFSET = FRONT_BALLS - 1
SEP_TOKEN = FRONT_BALLS + BACK_BALLS
VOCAB_SIZE = SEP_TOKEN + 1


def load_json_data(path):
    """Load lottery data from JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ball_to_token(ball, is_back=False):
    """Convert ball number to token ID"""
    if is_back:
        if not 1 <= ball <= BACK_BALLS:
            raise ValueError(f"Back ball out of range: {ball}")
        return BACK_TOKEN_OFFSET + ball  # 35-46 for back balls 1-12
    if not 1 <= ball <= FRONT_BALLS:
        raise ValueError(f"Front ball out of range: {ball}")
    return ball - 1  # 0-34 for front balls


def prepare_sequence(data):
    """
    Convert lottery history to token sequence
    Format: front1, front2, front3, front4, front5, back1, back2, sep, ...
    """
    tokens = []

    for draw in data:
        front = draw["frontBalls"]
        back = draw["backBalls"]
        if len(front) != 5 or len(set(front)) != 5:
            raise ValueError(f"Invalid front balls in issue {draw['issueNumber']}")
        if len(back) != 2 or len(set(back)) != 2:
            raise ValueError(f"Invalid back balls in issue {draw['issueNumber']}")

        for ball in sorted(front):
            tokens.append(ball_to_token(ball, is_back=False))

        for ball in sorted(back):
            tokens.append(ball_to_token(ball, is_back=True))

        tokens.append(SEP_TOKEN)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare lottery training data")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter data from this year onwards (e.g., 2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "lottery_train.bin",
        help="Output path for the chronological uint16 token stream",
    )
    parser.add_argument(
        "--holdout-draws",
        type=int,
        default=72,
        help="Keep this many latest draws out of the ANE training stream",
    )
    args = parser.parse_args()

    data_path = Path(__file__).parent / "data" / "dlt_merged.json"
    output_path = args.output

    print(f"Loading data from {data_path}...")
    data = load_json_data(data_path)
    data.sort(key=lambda draw: draw["date"])

    original_count = len(data)

    # Filter by year if specified
    if args.year:
        data = [d for d in data if int(d["date"][:4]) >= args.year]
        print(
            f"Filtered to year >= {args.year}: {len(data)} draws (was {original_count})"
        )
    else:
        print(f"Using all data: {len(data)} draws")

    if args.holdout_draws < 0 or len(data) <= args.holdout_draws:
        raise ValueError("Invalid holdout size for the available draws")

    training_data = (
        data[:-args.holdout_draws] if args.holdout_draws else data
    )
    print(
        f"Training range: {training_data[0]['date']} to "
        f"{training_data[-1]['date']} ({len(training_data)} draws)"
    )
    if args.holdout_draws:
        print(
            f"Holdout range: {data[-args.holdout_draws]['date']} to "
            f"{data[-1]['date']} ({args.holdout_draws} draws)"
        )

    tokens = prepare_sequence(training_data)
    if tokens and (min(tokens) < 0 or max(tokens) >= VOCAB_SIZE):
        raise ValueError("Prepared token is outside the configured vocabulary")
    print(f"Total tokens: {len(tokens)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack(f"<{len(tokens)}H", *tokens)
    output_path.write_bytes(payload)
    data_sha256 = hashlib.sha256(payload).hexdigest()
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest = {
        "format": "chronological_uint16_le_token_stream",
        "source": data_path.name,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_sha256": data_sha256,
        "source_draw_count": len(data),
        "draw_count": len(training_data),
        "holdout_draw_count": args.holdout_draws,
        "token_count": len(tokens),
        "date_from": training_data[0]["date"],
        "date_to": training_data[-1]["date"],
        "latest_issue": training_data[-1].get("issueNumber"),
        "holdout_date_from": (
            data[-args.holdout_draws]["date"] if args.holdout_draws else None
        ),
        "holdout_date_to": data[-1]["date"] if args.holdout_draws else None,
        "vocab_size": VOCAB_SIZE,
        "token_mapping": {
            "front": "ball - 1 (0-34)",
            "back": f"{BACK_TOKEN_OFFSET} + ball (35-46)",
            "separator": SEP_TOKEN,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved to {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")
    print(f"SHA-256: {data_sha256}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
