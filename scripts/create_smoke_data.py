from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    smoke_dir = Path("data/interim/smoke")

    train_rows = [
        {"text": "The company beat earnings expectations and raised guidance.", "label": "positive"},
        {"text": "Shares rallied after strong revenue growth in the quarter.", "label": "positive"},
        {"text": "The bank reported resilient margins and stable asset quality.", "label": "positive"},
        {"text": "Management maintained a cautious but unchanged outlook.", "label": "neutral"},
        {"text": "The board approved the proposal with limited market reaction.", "label": "neutral"},
        {"text": "Trading volume stayed flat and analyst estimates were unchanged.", "label": "neutral"},
        {"text": "The firm missed revenue expectations and lowered guidance.", "label": "negative"},
        {"text": "Credit losses widened and the stock fell sharply.", "label": "negative"},
        {"text": "Investors reacted negatively to the weak cash flow report.", "label": "negative"},
    ]

    in_domain_rows = [
        {"text": "Quarterly profit improved and the shares advanced.", "label": "positive"},
        {"text": "Results were mostly in line with expectations.", "label": "neutral"},
        {"text": "The company warned about slower sales next quarter.", "label": "negative"},
    ]

    cross_dataset_rows = [
        {"text": "Broker upgrades pushed the stock higher in early trading.", "label": "bullish"},
        {"text": "Market participants stayed on the sidelines awaiting new data.", "label": "neutral"},
        {"text": "The issuer faced bearish sentiment after another weak forecast.", "label": "bearish"},
    ]

    write_split(smoke_dir / "train_dataset_a.csv", train_rows)
    write_split(smoke_dir / "test_dataset_a.csv", in_domain_rows)
    write_split(smoke_dir / "test_dataset_b.csv", cross_dataset_rows)

    print(f"Smoke data written to: {smoke_dir.resolve()}")


if __name__ == "__main__":
    main()
