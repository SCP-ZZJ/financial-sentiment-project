from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.dataset import CANONICAL_LABELS


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = list(CANONICAL_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_confusion_matrix_frame(
    y_true: list[str], y_pred: list[str]
) -> pd.DataFrame:
    labels = list(CANONICAL_LABELS)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)
