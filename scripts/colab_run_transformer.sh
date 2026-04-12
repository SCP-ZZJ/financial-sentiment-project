#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/transformer_finbert_fpb.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

python -m src.cli.train_transformer --config "${CONFIG_PATH}"
