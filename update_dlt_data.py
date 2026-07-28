#!/usr/bin/env python3
"""Rebuild the DLT history only after independent sources agree exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dlt_merged.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "dlt_merged.manifest.json"
DEFAULT_GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "yangxb919/lottery-data/main/data/dlt.json"
)
DEFAULT_GITHUB_COMMIT_URL = (
    "https://api.github.com/repos/yangxb919/lottery-data/commits/main"
)
DEFAULT_GITHUB_REPOSITORY = (
    "https://github.com/yangxb919/lottery-data.git"
)
DEFAULT_GITHUB_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/"
    "Hsxzn/sportsLottery/"
    "ca1d336d75f81e7a421f5a81c9baa04586f46538/"
    "src/sportLottery/dataConfig/all.json"
)
DEFAULT_OFFICIAL_URL = (
    "https://webapi.sporttery.cn/gateway/lottery/"
    "getHistoryPageListV1.qry"
)
ISSUE_PATTERN = re.compile(r"^\d{5}$")


class DataValidationError(RuntimeError):
    """Raised when a source cannot prove a complete, consistent dataset."""


def build_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.sporttery.cn",
            "Referer": "https://www.sporttery.cn/",
            "User-Agent": "lottery-prediction-data-audit/1.0",
        }
    )
    return session


def fetch_json(
    session: requests.Session, url: str, timeout: float, params: dict[str, Any] | None = None
) -> Any:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def normalize_draw(
    issue: Any,
    draw_date: Any,
    front: Iterable[Any],
    back: Iterable[Any],
) -> dict[str, Any]:
    return {
        "issueNumber": str(issue).zfill(5),
        "date": str(draw_date),
        "frontBalls": [int(value) for value in front],
        "backBalls": [int(value) for value in back],
    }


def normalize_github(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise DataValidationError("GitHub full-history payload is not a list")
    return [
        normalize_draw(
            row.get("issue"),
            row.get("date"),
            row.get("front", ()),
            row.get("back", ()),
        )
        for row in data
    ]


def normalize_github_snapshot(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise DataValidationError("GitHub official snapshot payload is not an object")

    draws: list[dict[str, Any]] = []
    for rows in data.values():
        if not isinstance(rows, list):
            raise DataValidationError("GitHub official snapshot contains a non-list year")
        for row in rows:
            result = str(row.get("lotteryDrawResult", "")).split()
            draws.append(
                normalize_draw(
                    row.get("lotteryDrawNum"),
                    row.get("lotteryDrawTime"),
                    result[:5],
                    result[5:],
                )
            )
    return draws


def normalize_official(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    draws: list[dict[str, Any]] = []
    for row in rows:
        result = str(row.get("lotteryDrawResult", "")).split()
        draws.append(
            normalize_draw(
                row.get("lotteryDrawNum"),
                row.get("lotteryDrawTime"),
                result[:5],
                result[5:],
            )
        )
    return draws


def validate_draws(
    draws: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    if not draws:
        raise DataValidationError(f"{label}: no draws returned")

    by_issue: dict[str, dict[str, Any]] = {}
    parsed_dates: dict[str, date] = {}
    for draw in draws:
        issue = draw["issueNumber"]
        if not ISSUE_PATTERN.fullmatch(issue):
            raise DataValidationError(f"{label}: malformed issue number {issue!r}")
        if issue in by_issue:
            raise DataValidationError(f"{label}: duplicate issue {issue}")

        try:
            parsed_date = date.fromisoformat(draw["date"])
        except ValueError as exc:
            raise DataValidationError(
                f"{label}: invalid date for issue {issue}: {draw['date']!r}"
            ) from exc
        if parsed_date > date.today():
            raise DataValidationError(f"{label}: future date for issue {issue}")
        if issue[:2] != f"{parsed_date.year % 100:02d}":
            raise DataValidationError(
                f"{label}: issue/date year mismatch at {issue} ({draw['date']})"
            )

        front = draw["frontBalls"]
        back = draw["backBalls"]
        if len(front) != 5 or front != sorted(set(front)):
            raise DataValidationError(f"{label}: invalid front set at issue {issue}")
        if len(back) != 2 or back != sorted(set(back)):
            raise DataValidationError(f"{label}: invalid back set at issue {issue}")
        if any(value < 1 or value > 35 for value in front):
            raise DataValidationError(f"{label}: front number out of range at {issue}")
        if any(value < 1 or value > 12 for value in back):
            raise DataValidationError(f"{label}: back number out of range at {issue}")

        by_issue[issue] = draw
        parsed_dates[issue] = parsed_date

    per_year: dict[str, list[int]] = defaultdict(list)
    for issue in by_issue:
        per_year[issue[:2]].append(int(issue[2:]))
    for year, sequences in sorted(per_year.items()):
        actual = sorted(sequences)
        expected = list(range(1, actual[-1] + 1))
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            raise DataValidationError(
                f"{label}: non-contiguous issues in {year}, missing {missing[:10]}"
            )

    ordered = sorted(draws, key=lambda draw: (draw["date"], draw["issueNumber"]))
    for previous, current in zip(ordered, ordered[1:]):
        if parsed_dates[current["issueNumber"]] <= parsed_dates[previous["issueNumber"]]:
            raise DataValidationError(
                f"{label}: dates are not strictly increasing at "
                f"{previous['issueNumber']} -> {current['issueNumber']}"
            )
    return ordered


def compare_exact(
    expected: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    expected_label: str,
    candidate_label: str,
) -> None:
    expected_by_issue = {draw["issueNumber"]: draw for draw in expected}
    candidate_by_issue = {draw["issueNumber"]: draw for draw in candidate}
    issues = sorted(set(expected_by_issue) | set(candidate_by_issue))
    differences = [
        issue
        for issue in issues
        if expected_by_issue.get(issue) != candidate_by_issue.get(issue)
    ]
    if differences:
        preview = ", ".join(differences[:10])
        raise DataValidationError(
            f"{expected_label} and {candidate_label} disagree at "
            f"{len(differences)} issue(s): {preview}"
        )


def fetch_official_history(
    session: requests.Session,
    url: str,
    timeout: float,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    base_params = {
        "gameNo": "85",
        "provinceId": "0",
        "pageSize": str(page_size),
        "isVerify": "1",
    }
    first_page = fetch_json(
        session, url, timeout, {**base_params, "pageNo": "1"}
    )
    value = validate_official_page(first_page, 1)
    pages = int(value["pages"])
    total = int(value["total"])
    rows = list(value["list"])

    for page_number in range(2, pages + 1):
        payload = fetch_json(
            session,
            url,
            timeout,
            {**base_params, "pageNo": str(page_number)},
        )
        page_value = validate_official_page(payload, page_number)
        if int(page_value["total"]) != total or int(page_value["pages"]) != pages:
            raise DataValidationError(
                f"official API pagination changed while reading page {page_number}"
            )
        rows.extend(page_value["list"])

    if len(rows) != total:
        raise DataValidationError(
            f"official API returned {len(rows)} rows but declared {total}"
        )
    return rows, pages


def validate_official_page(payload: Any, page_number: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise DataValidationError(f"official API rejected page {page_number}")
    value = payload.get("value")
    if not isinstance(value, dict) or not isinstance(value.get("list"), list):
        raise DataValidationError(f"official API malformed page {page_number}")
    if int(value.get("pageNo", 0)) != page_number:
        raise DataValidationError(
            f"official API returned page {value.get('pageNo')} for request {page_number}"
        )
    return value


def canonical_bytes(draws: list[dict[str, Any]]) -> bytes:
    text = json.dumps(draws, ensure_ascii=False, indent=2) + "\n"
    return text.encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def github_commit_sha(
    session: requests.Session,
    url: str,
    repository: str,
    timeout: float,
) -> str | None:
    try:
        payload = fetch_json(session, url, timeout)
    except (requests.RequestException, ValueError):
        payload = None
    sha = payload.get("sha") if isinstance(payload, dict) else None
    if sha:
        return str(sha)

    try:
        completed = subprocess.run(
            ["git", "ls-remote", repository, "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    fields = completed.stdout.split()
    return fields[0] if fields else None


def source_host(url: str) -> str:
    return urlparse(url).hostname or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild DLT history after a GitHub full-history dataset and "
            "the official Sporttery API agree exactly."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--github-url", default=DEFAULT_GITHUB_URL)
    parser.add_argument(
        "--github-commit-url", default=DEFAULT_GITHUB_COMMIT_URL
    )
    parser.add_argument(
        "--github-repository", default=DEFAULT_GITHUB_REPOSITORY
    )
    parser.add_argument(
        "--github-snapshot-url", default=DEFAULT_GITHUB_SNAPSHOT_URL
    )
    parser.add_argument("--official-url", default=DEFAULT_OFFICIAL_URL)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="number of latest calendar years to keep for model training",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate all sources without changing local files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_size < 1 or args.page_size > 100:
        raise SystemExit("--page-size must be between 1 and 100")
    if args.timeout <= 0 or args.retries < 0:
        raise SystemExit("--timeout must be positive and --retries non-negative")
    if args.years < 1:
        raise SystemExit("--years must be positive")

    session = build_session(args.retries)

    github_payload = fetch_json(session, args.github_url, args.timeout)
    github_draws = validate_draws(
        normalize_github(github_payload), "GitHub full history"
    )

    snapshot_payload = fetch_json(
        session, args.github_snapshot_url, args.timeout
    )
    snapshot_draws = validate_draws(
        normalize_github_snapshot(snapshot_payload),
        "GitHub official snapshot",
    )

    official_rows, official_pages = fetch_official_history(
        session,
        args.official_url,
        args.timeout,
        args.page_size,
    )
    official_draws = validate_draws(
        normalize_official(official_rows), "official Sporttery API"
    )

    compare_exact(
        official_draws,
        github_draws,
        "official Sporttery API",
        "GitHub full history",
    )
    official_by_issue = {
        draw["issueNumber"]: draw for draw in official_draws
    }
    official_snapshot_slice = [
        official_by_issue[draw["issueNumber"]]
        for draw in snapshot_draws
        if draw["issueNumber"] in official_by_issue
    ]
    compare_exact(
        official_snapshot_slice,
        snapshot_draws,
        "official Sporttery API",
        "GitHub official snapshot",
    )

    latest_date = date.fromisoformat(official_draws[-1]["date"])
    first_year = latest_date.year - args.years + 1
    training_draws = [
        draw
        for draw in official_draws
        if date.fromisoformat(draw["date"]).year >= first_year
    ]
    if not training_draws:
        raise DataValidationError("calendar-year training window produced no draws")

    output_content = canonical_bytes(training_draws)
    data_sha256 = sha256_bytes(output_content)
    github_sources = sorted(
        {
            str(row.get("source"))
            for row in github_payload
            if isinstance(row, dict) and row.get("source")
        }
    )
    github_commit = github_commit_sha(
        session,
        args.github_commit_url,
        args.github_repository,
        args.timeout,
    )
    if (
        github_commit is None
        and args.output.exists()
        and args.output.read_bytes() == output_content
        and args.manifest.exists()
    ):
        try:
            existing_manifest = json.loads(
                args.manifest.read_text(encoding="utf-8")
            )
            existing_commit = existing_manifest["sources"][
                "githubFullHistory"
            ]["commit"]
        except (OSError, ValueError, KeyError, TypeError):
            existing_commit = None
        if existing_commit:
            github_commit = str(existing_commit)

    manifest = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "verification": {
            "result": "exact_match",
            "fullHistoryComparedRecords": len(official_draws),
            "officialSnapshotComparedRecords": len(snapshot_draws),
            "duplicateIssues": 0,
            "missingIssuesWithinYears": 0,
        },
        "window": {
            "type": "calendar_years",
            "years": args.years,
            "fromYearInclusive": first_year,
            "throughYear": latest_date.year,
        },
        "dataset": {
            "records": len(training_draws),
            "firstIssue": training_draws[0]["issueNumber"],
            "firstDate": training_draws[0]["date"],
            "latestIssue": training_draws[-1]["issueNumber"],
            "latestDate": training_draws[-1]["date"],
            "sha256": data_sha256,
        },
        "sources": {
            "githubFullHistory": {
                "url": args.github_url,
                "host": source_host(args.github_url),
                "commit": github_commit,
                "declaredUpstream": github_sources,
            },
            "githubOfficialSnapshot": {
                "url": args.github_snapshot_url,
                "host": source_host(args.github_snapshot_url),
                "records": len(snapshot_draws),
            },
            "officialSportteryApi": {
                "url": args.official_url,
                "host": source_host(args.official_url),
                "gameNo": "85",
                "pages": official_pages,
                "records": len(official_draws),
            },
        },
    }
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    print(
        f"Verified full history of {len(official_draws)} draws: "
        f"{official_draws[0]['issueNumber']} ({official_draws[0]['date']}) -> "
        f"{official_draws[-1]['issueNumber']} ({official_draws[-1]['date']})"
    )
    print(
        "Exact source matches: "
        f"full={len(official_draws)}, snapshot={len(snapshot_draws)}"
    )
    print(
        f"Latest {args.years} calendar years: {len(training_draws)} draws, "
        f"{training_draws[0]['issueNumber']} ({training_draws[0]['date']}) -> "
        f"{training_draws[-1]['issueNumber']} ({training_draws[-1]['date']})"
    )
    print(f"Dataset SHA-256: {data_sha256}")

    if args.check_only:
        print("Check-only mode: local files were not changed")
        return 0

    if args.output.exists():
        existing = args.output.read_bytes()
        if existing != output_content:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = args.output.with_name(
                f"{args.output.stem}.before-{timestamp}{args.output.suffix}"
            )
            atomic_write(backup, existing)
            print(f"Previous dataset backed up to {backup}")

    output_changed = (
        not args.output.exists() or args.output.read_bytes() != output_content
    )
    manifest_changed = True
    if args.manifest.exists():
        try:
            existing_manifest = json.loads(
                args.manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            existing_manifest = None
        if isinstance(existing_manifest, dict):
            comparable_existing = {
                key: value
                for key, value in existing_manifest.items()
                if key != "generatedAt"
            }
            comparable_new = {
                key: value
                for key, value in manifest.items()
                if key != "generatedAt"
            }
            if comparable_existing == comparable_new:
                manifest_changed = False

    if output_changed:
        atomic_write(args.output, output_content)
        print(f"Wrote {args.output}")
    else:
        print(f"Dataset unchanged: {args.output}")
    if manifest_changed:
        atomic_write(args.manifest, manifest_content)
        print(f"Wrote {args.manifest}")
    else:
        print(f"Manifest unchanged: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
