# ECG-Explain

A 12-lead ECG classifier that surfaces *why* it predicts what it predicts — per-lead Grad-CAM overlays highlighting the waveform regions driving each diagnosis.

**Status:** in development.

## Motivation

Coming soon — clinical rationale from a foundation doctor's perspective.

## Dataset

PTB-XL — 21,837 12-lead ECGs across 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).

## Model

1D ResNet, multi-label classification.

## Interpretability

1D Grad-CAM with per-lead attribution overlays.

## Reproducing

Install dependencies (one-time):

    uv sync --all-extras

Download PTB-XL (~1 GB, ~30 min on a typical home connection):

    uv run python scripts/download_data.py

Quick smoke test (2 epochs of a small model, ~3 min):

    uv run python scripts/train.py configs/smoke.yaml

Full training run:

    uv run python scripts/train.py configs/baseline.yaml

Evaluate on the test set:

    uv run python scripts/evaluate.py \
        --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline/best.pt \
        --output results/baseline_test_metrics.json

Run the test suite:

    uv run pytest

## License

MIT.