# Financial Sentiment Analysis Project

## Goal
This project studies financial sentiment classification with two datasets under a unified 3-class setup:
- positive
- neutral
- negative

The planned experiments include:
- in-domain evaluation
- cross-dataset evaluation
- TF-IDF + Logistic Regression baseline
- FinBERT / BERT-based fine-tuning
- optional domain-adaptive pretraining (DAPT)

## Project Structure
- `configs/`: experiment configs
- `data/raw`, `data/interim`, `data/processed`: dataset storage by stage
- `src/data`: data loading and label normalization
- `src/models`: baseline and future transformer code
- `src/evaluation`: metrics and confusion matrix helpers
- `src/cli`: experiment entrypoints
- `outputs/metrics`, `outputs/predictions`, `outputs/logs`, `outputs/checkpoints`: run artifacts
- `scripts/`: smoke test and Colab helper scripts

## Workflow
The intended workflow is:
- write and version code locally
- keep experiment logic in `src/` and `configs/`
- use Colab as an execution environment instead of writing core logic inside notebooks
- save metrics, predictions, and checkpoints back to persistent storage

This keeps the codebase reproducible while still using free GPU access when needed.

## Current Baseline
The repository now includes a minimal runnable baseline:
- YAML-based experiment config
- CSV dataset loader with label mapping into `positive` / `neutral` / `negative`
- TF-IDF + Logistic Regression training pipeline
- evaluation output for multiple named test sets
- prediction CSV export
- confusion matrix CSV export
- pickled model checkpoint export

## Expected Data Format
The current baseline expects CSV files with at least:
- a text column, default `text`
- a label column, default `label`

The label values can be mapped through `data.label_map` in the config, for example:
- `bullish -> positive`
- `bearish -> negative`

## Key Commands
Run the smoke test locally:

```bash
python scripts/run_smoke_test.py
```

Run the baseline with a config:

```bash
python -m src.cli.train_baseline --config configs/baseline.yaml
```

Generate the toy smoke-test data only:

```bash
python scripts/create_smoke_data.py
```

## Colab Usage
Recommended Colab pattern:

1. Clone the repo.
2. Install dependencies.
3. Mount Google Drive if you want persistent outputs.
4. Run the same project scripts you use locally.

Example Colab cells:

```bash
!git clone <your-repo-url>
%cd financial-sentiment-project
!bash scripts/colab_setup.sh
!python scripts/run_smoke_test.py
```

Or run the real baseline:

```bash
!bash scripts/colab_run_baseline.sh configs/baseline.yaml
```

## Statistical Baselines

Run all statistical baselines (LogReg, Linear SVM, Naive Bayes) on both datasets:

```bash
python scripts/run_statistical_baselines.py
```

Results are saved to `outputs/metrics/statistical_baselines/summary.csv`.

## Transformer Baselines

Fine-tune BERT-base or FinBERT on a single config:

```bash
python -m src.cli.train_transformer --config configs/transformer_finbert_fpb.yaml
```

Run all four transformer baselines (BERT×2 + FinBERT×2):

```bash
python scripts/run_transformer_baselines.py
```

Results are saved to `outputs/metrics/transformer_baselines/summary.csv`.

## Continued MLM Pretraining + Fine-tuning

Step 1: Run continued MLM pretraining on in-domain financial text:

```bash
python -m src.cli.run_mlm_pretrain --config configs/mlm_pretrain.yaml
```

Step 2: Update the `model.name` field in `configs/transformer_finbert_mlm_fpb.yaml` (or `_tfns.yaml`) to point to the saved MLM checkpoint directory.

Step 3: Fine-tune:

```bash
python -m src.cli.train_transformer --config configs/transformer_finbert_mlm_fpb.yaml
```

## Colab Usage

Recommended Colab pattern:

```bash
!git clone <your-repo-url>
%cd financial-sentiment-project
!bash scripts/colab_setup.sh

# Statistical baselines (CPU is fine)
!python scripts/run_statistical_baselines.py

# Transformer fine-tuning (GPU recommended)
!python -m src.cli.train_transformer --config configs/transformer_finbert_fpb.yaml

# Or run all four transformer baselines
!python scripts/run_transformer_baselines.py

# Continued MLM pretraining
!python -m src.cli.run_mlm_pretrain --config configs/mlm_pretrain.yaml

# Then fine-tune from the MLM checkpoint
!python -m src.cli.train_transformer --config configs/transformer_finbert_mlm_fpb.yaml
```

To persist outputs to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs /content/drive/MyDrive/financial-sentiment-outputs
```