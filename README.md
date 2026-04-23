# Financial Sentiment Analysis

## Overview
This project studies financial sentiment classification under a unified 3-class setup:
- positive
- neutral
- negative

Implemented pipelines include:
- statistical baselines: TF-IDF + Logistic Regression, Linear SVM, Naive Bayes
- lexicon baseline: Loughran-McDonald rule-based classifier
- transformer fine-tuning: BERT and FinBERT
- optional continued MLM pretraining before fine-tuning for FinBERT and BERT

The project supports both in-domain and cross-dataset evaluation.

## Repository Structure
- `configs/`: experiment configuration files
- `data/raw`, `data/interim`, `data/processed`: datasets by processing stage
- `src/data`: loading, preprocessing, and label normalization
- `src/models`: statistical, rule-based, transformer, and MLM models
- `src/evaluation`: metrics and plotting helpers
- `src/cli`: experiment entrypoints
- `scripts/`: smoke tests, batch runners, and helper scripts
- `outputs/`: generated metrics, predictions, figures, logs, checkpoints, and EDA artifacts
- `notebooks/`: exploratory notebooks

## Setup
Recommended environment:
- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format
Training and evaluation files are expected to be CSV files with at least:
- a text column, default `text`
- a label column, default `label`

Label aliases can be normalized through `data.label_map` in each config.

## Quick Start
Run a minimal end-to-end smoke test:

```bash
python scripts/run_smoke_test.py
```

Generate the smoke-test dataset only:

```bash
python scripts/create_smoke_data.py
```

## Main Commands
Run one statistical baseline:

```bash
python -m src.cli.train_baseline --config configs/baseline.yaml
```

Run all statistical baselines on both datasets:

```bash
python scripts/run_statistical_baselines.py
```

Run the rule-based baseline:

```bash
python -m src.cli.run_rule_based_baseline --config configs/rule_based.yaml
```

Run one transformer experiment:

```bash
python -m src.cli.train_transformer --config configs/transformer_finbert_fpb.yaml
```

Run all transformer baselines:

```bash
python scripts/run_transformer_baselines.py
```

Run continued MLM pretraining:

```bash
python -m src.cli.run_mlm_pretrain --config configs/mlm_pretrain.yaml
python -m src.cli.run_mlm_pretrain --config configs/mlm_pretrain_bert.yaml
```

Then fine-tune from the saved MLM checkpoint by updating `model.name` in one of:
- `configs/transformer_finbert_mlm_fpb.yaml`
- `configs/transformer_finbert_mlm_tfns.yaml`
- `configs/transformer_bert_mlm_fpb.yaml`
- `configs/transformer_bert_mlm_tfns.yaml`

Optional BERT augmentation configs are also included:
- `configs/mlm_pretrain_bert_augmentation.yaml`
- `configs/transformer_bert_mlm_fpb_augmentation.yaml`
- `configs/transformer_bert_mlm_tfns_augmentation.yaml`

## Outputs
Key generated outputs include:
- `outputs/metrics/`: JSON summaries, confusion matrices, and batch summaries
- `outputs/predictions/`: prediction CSV files
- `outputs/figures/`: confusion matrices and training curves
- `outputs/logs/`: config snapshots and trainer logs
- `outputs/checkpoints/`: saved statistical models and transformer checkpoints
- `outputs/eda/`: exploratory data analysis artifacts

Batch summaries are written to:
- `outputs/metrics/statistical_baselines/summary.csv`
- `outputs/metrics/transformer_baselines/summary.csv`

## Colab
For Colab, the recommended workflow is:

```bash
!git clone <your-repo-url>
%cd financial-sentiment-project
!bash scripts/colab_setup.sh
!python scripts/run_smoke_test.py
```

For larger runs:
- statistical baselines can run on CPU
- transformer fine-tuning and MLM pretraining are better with GPU

To persist outputs to Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r outputs /content/drive/MyDrive/financial-sentiment-outputs
```