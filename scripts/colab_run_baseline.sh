#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baseline.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

python -m src.cli.train_baseline --config "${CONFIG_PATH}"
