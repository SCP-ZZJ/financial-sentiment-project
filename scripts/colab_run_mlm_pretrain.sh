#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/mlm_pretrain.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

python -m src.cli.run_mlm_pretrain --config "${CONFIG_PATH}"
