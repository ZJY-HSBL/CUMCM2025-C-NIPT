#!/usr/bin/env python3
"""Run the unmodified submission scripts with local non-public data."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {1: "a_1_1.xlsx", 2: "a_1.xlsx", 3: "附件.xlsx", 4: "nvtai.xlsx"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "private_data")
    parser.add_argument("--no-gui", action="store_true", help="Use noninteractive Matplotlib backend")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    need = data_dir / REQUIRED[args.question]
    if not need.is_file():
        parser.error(f"Required local input not found: {need}")
    script = ROOT / "src" / "original" / f"question_{args.question}.py"
    env = os.environ.copy()
    if args.no_gui:
        env["MPLBACKEND"] = "Agg"
    print(f"Running original question {args.question} from {data_dir}", flush=True)
    return subprocess.call([sys.executable, str(script)], cwd=data_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
