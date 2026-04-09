from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_pipeline(model_config: dict[str, Any]) -> Pipeline:
    ngram_range = tuple(model_config.get("ngram_range", [1, 2]))

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    max_features=model_config.get("max_features"),
                    min_df=model_config.get("min_df", 1),
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=model_config.get("C", 1.0),
                    max_iter=model_config.get("max_iter", 300),
                    class_weight=model_config.get("class_weight"),
                ),
            ),
        ]
    )
