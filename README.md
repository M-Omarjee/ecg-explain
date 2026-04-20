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

    uv sync --all-extras
    uv run pytest

## License

MIT.