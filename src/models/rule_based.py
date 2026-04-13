from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.dataset import CANONICAL_LABELS
from src.utils.paths import resolve_path


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass
class RuleBasedPrediction:
    prediction: str
    positive_hits: int
    negative_hits: int
    score: int


class LoughranMcDonaldRuleBaseline:
    def __init__(
        self,
        positive_words: set[str],
        negative_words: set[str],
        min_count_diff: int = 1,
    ) -> None:
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.min_count_diff = min_count_diff

    @classmethod
    def from_csv(
        cls,
        path_str: str,
        positive_column: str = "Positive",
        negative_column: str = "Negative",
        word_column: str = "Word",
        lexicon_threshold: int = 0,
        min_count_diff: int = 1,
    ) -> "LoughranMcDonaldRuleBaseline":
        path = resolve_path(path_str)
        lexicon = pd.read_csv(path)

        positive_words = set(
            lexicon.loc[lexicon[positive_column] > lexicon_threshold, word_column]
            .astype(str)
            .str.upper()
        )
        negative_words = set(
            lexicon.loc[lexicon[negative_column] > lexicon_threshold, word_column]
            .astype(str)
            .str.upper()
        )
        return cls(
            positive_words=positive_words,
            negative_words=negative_words,
            min_count_diff=min_count_diff,
        )

    def predict_text(self, text: str) -> RuleBasedPrediction:
        tokens = [token.upper() for token in TOKEN_PATTERN.findall(text)]
        positive_hits = sum(token in self.positive_words for token in tokens)
        negative_hits = sum(token in self.negative_words for token in tokens)
        score = positive_hits - negative_hits

        if score >= self.min_count_diff:
            label = "positive"
        elif score <= -self.min_count_diff:
            label = "negative"
        else:
            label = "neutral"

        return RuleBasedPrediction(
            prediction=label,
            positive_hits=int(positive_hits),
            negative_hits=int(negative_hits),
            score=int(score),
        )

    def predict_frame(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            prediction = self.predict_text(text)
            rows.append(
                {
                    "prediction": prediction.prediction,
                    "positive_hits": prediction.positive_hits,
                    "negative_hits": prediction.negative_hits,
                    "lexicon_score": prediction.score,
                }
            )
        return pd.DataFrame(rows)
