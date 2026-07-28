#!/usr/bin/env python3
"""Install, remove, inspect, or render the macOS launchd schedule."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "automation" / "config.json"
LABEL = "com.opendown.lottery-prediction"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def launch_domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist() -> dict[str, Any]:
    config = load_config()
    log_directory = PROJECT_ROOT / "ane_training" / "scheduler"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(PROJECT_ROOT / "automation" / "run_scheduled_cycle.sh")
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": [
            {
                "Hour": int(item["hour"]),
                "Minute": int(item["minute"]),
            }
            for item in config["schedule"]
        ],
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": str(log_directory / "stdout.log"),
        "StandardErrorPath": str(log_directory / "stderr.log"),
        "EnvironmentVariables": {
            "PATH": (
                "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:"
                "/usr/sbin:/sbin"
            ),
            "PYTHONUNBUFFERED": "1",
        },
    }


def render() -> bytes:
    return plistlib.dumps(build_plist(), sort_keys=True)


def install() -> None:
    destination = launch_agent_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "ane_training" / "scheduler").mkdir(
        parents=True, exist_ok=True
    )
    if destination.exists():
        subprocess.run(
            [
                "/bin/launchctl",
                "bootout",
                launch_domain(),
                str(destination),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    destination.write_bytes(render())
    subprocess.run(
        [
            "/bin/launchctl",
            "bootstrap",
            launch_domain(),
            str(destination),
        ],
        check=True,
    )
    print(f"Installed and loaded {destination}")


def uninstall() -> None:
    destination = launch_agent_path()
    if destination.exists():
        subprocess.run(
            [
                "/bin/launchctl",
                "bootout",
                launch_domain(),
                str(destination),
            ],
            check=False,
        )
        destination.unlink()
        print(f"Removed {destination}")
    else:
        print(f"Schedule is not installed: {destination}")


def status() -> int:
    completed = subprocess.run(
        ["/bin/launchctl", "print", f"{launch_domain()}/{LABEL}"],
        check=False,
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("install", "uninstall", "status", "print")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "install":
        install()
    elif args.action == "uninstall":
        uninstall()
    elif args.action == "status":
        return status()
    else:
        print(render().decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
