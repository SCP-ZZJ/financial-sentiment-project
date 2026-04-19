# Cross-Dataset Comparison

## Label Balance
- FPB is smaller (2264 rows) and still noticeably imbalanced, with `neutral` dominating.
- TFNS is much larger (11931 rows) and heavily skewed toward label `2`, while labels `0` and `1` are much less frequent.

## Text Length
- FPB average text length: 22.4417 words, 121.9625 characters.
- TFNS average text length: 12.207 words, 86.0215 characters.
- This suggests FPB contains more sentence-like financial statements, while TFNS is closer to short headlines or alert-style snippets.

## Writing Style
- FPB rows with URLs / tickers: 19 / 0.
- TFNS rows with URLs / tickers: 5612 / 1804.
- TFNS contains much more platform-style noise such as URLs, ticker symbols, and short brokerage-action headlines.

## Data Quality Risks
- FPB has replacement-character rows: 47, indicating encoding cleanup may be needed later.
- TFNS appears structurally clean, but the numeric labels should be verified against the dataset documentation before any later label mapping step.

## Cross-Dataset Evaluation Risks
- Domain shift is likely to be substantial because the two datasets differ in text length, writing style, and label format.
- URL/ticker-heavy TFNS text may cause models trained on FPB to underperform on TFNS unless tokenization and preprocessing are handled carefully in a later step.
- Severe class imbalance, especially in TFNS, may distort macro-F1 and cross-dataset transfer unless addressed later during training or sampling.
