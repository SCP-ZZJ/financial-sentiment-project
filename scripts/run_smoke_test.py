from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    run_step([sys.executable, "scripts/create_smoke_data.py"])
    run_step([sys.executable, "-m", "src.cli.train_baseline", "--config", "configs/smoke_baseline.yaml"])
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
