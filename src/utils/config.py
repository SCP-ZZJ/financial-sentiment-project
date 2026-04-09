from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import resolve_path


def load_yaml_config(path_str: str) -> dict[str, Any]:
    config_path = resolve_path(path_str)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")

    config["_config_path"] = str(config_path)
    return config


def dump_yaml_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
