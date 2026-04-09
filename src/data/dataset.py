from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.paths import resolve_path


CANONICAL_LABELS = ("positive", "neutral", "negative")
DEFAULT_LABEL_MAP = {
    "positive": "positive",
    "pos": "positive",
    "bullish": "positive",
    "1": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "mixed": "neutral",
    "0": "neutral",
    "negative": "negative",
    "neg": "negative",
    "bearish": "negative",
    "-1": "negative",
}


def _build_label_map(configured_map: dict[str, Any] | None) -> dict[str, str]:
    merged = dict(DEFAULT_LABEL_MAP)
    if configured_map:
        merged.update({str(key).strip().lower(): str(value).strip().lower() for key, value in configured_map.items()})
    return merged


def _normalize_label(raw_label: Any, label_map: dict[str, str]) -> str:
    normalized_key = str(raw_label).strip().lower()
    mapped = label_map.get(normalized_key)
    if mapped is None:
        raise ValueError(f"Unsupported label '{raw_label}'. Add it to data.label_map.")
    if mapped not in CANONICAL_LABELS:
        raise ValueError(
            f"Mapped label '{mapped}' must be one of {CANONICAL_LABELS}."
        )
    return mapped


def load_labeled_csv(
    path_str: str,
    text_column: str = "text",
    label_column: str = "label",
    label_map: dict[str, Any] | None = None,
) -> pd.DataFrame:
    path = resolve_path(path_str)
    frame = pd.read_csv(path)
    normalized_label_map = _build_label_map(label_map)

    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    prepared = frame[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    prepared = prepared.dropna(subset=["text", "label"]).copy()
    prepared["text"] = prepared["text"].astype(str).str.strip()
    prepared["label"] = prepared["label"].map(
        lambda item: _normalize_label(item, normalized_label_map)
    )
    prepared = prepared[prepared["text"] != ""].reset_index(drop=True)

    if prepared.empty:
        raise ValueError(f"{path} produced an empty dataset after preprocessing.")

    return prepared
