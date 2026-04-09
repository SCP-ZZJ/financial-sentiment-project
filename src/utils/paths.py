from __future__ import annotations

import os
from pathlib import Path


PROJECT_SENTINELS = ("README.md", "requirements.txt")


def get_project_root() -> Path:
    """Resolve the repository root locally or inside Colab."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if all((parent / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return parent

    raise FileNotFoundError(
        "Could not locate project root. Set PROJECT_ROOT or run from the repo."
    )


def resolve_path(path_str: str | os.PathLike[str]) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def ensure_dir(path_str: str | os.PathLike[str]) -> Path:
    path = resolve_path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path
