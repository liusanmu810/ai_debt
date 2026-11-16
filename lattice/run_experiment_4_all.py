"""
Run all three Experiment 4 heatmap data scripts in sequence.

This wrapper is intended for PyCharm or command-line execution:

    python run_experiment_4_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    "experiment_4_A_wM_data.py",
    "experiment_4_A_alpha_data.py",
    "experiment_4_wM_alpha_data.py",
]


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> None:
    root = Path(__file__).resolve().parent
    total_start = time.time()

    print("\n================ Experiment 4 Batch Run ================")
    print(f"Working dir : {root}")
    print(f"Python      : {sys.executable}")
    print(f"Scripts     : {len(SCRIPTS)}")
    print("========================================================\n")

    for index, script_name in enumerate(SCRIPTS, start=1):
        script_path = root / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        print("--------------------------------------------------------")
        print(f"[{index}/{len(SCRIPTS)}] Running: {script_name}")
        print("--------------------------------------------------------")

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=str(root),
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(
                f"{script_name} failed with exit code {result.returncode} "
                f"after {format_seconds(elapsed)}."
            )

        print(f"\nFinished: {script_name}")
        print(f"Elapsed : {format_seconds(elapsed)}\n")

    total_elapsed = time.time() - total_start
    print("================ Experiment 4 Batch Finished ================")
    print(f"Total time: {format_seconds(total_elapsed)}")
    print("=============================================================")


if __name__ == "__main__":
    main()
