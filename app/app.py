"""ECG-Explain — Gradio demo.

A browser-based interface for the trained ECG classifier. Users can:
    - Pick from a curated set of test records, or upload their own
    - See predicted class probabilities
    - View a Grad-CAM overlay highlighting which waveform regions drove
      the prediction for a chosen class

Run locally:
    uv sync --all-extras
    uv run python app/app.py

Deploy: this is the entry point Hugging Face Spaces will run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless backend, required for Spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb

# Make `ecg_explain` importable when the app is run from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg_explain.data.labels import SUPERCLASS_TO_IDX, SUPERCLASSES  # noqa: E402
from ecg_explain.data.preprocessing import preprocess_ecg  # noqa: E402
from ecg_explain.interpret import GradCAM1D  # noqa: E402
from ecg_explain.models import resnet1d_medium  # noqa: E402
from ecg_explain.viz import plot_prediction_summary  # noqa: E402

# --- Config (override with env vars on Spaces) ---

# Local path for the checkpoint. If absent, we try to download from HF Hub.
CHECKPOINT_PATH = Path(os.environ.get("CKPT_PATH", "checkpoints/baseline/best.pt"))

# Hugging Face Hub model repo (set this once you've uploaded weights):
#   format "username/repo-name", e.g. "M-Omarjee/ecg-explain-resnet1d"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_CKPT_FILENAME = os.environ.get("HF_CKPT_FILENAME", "best.pt")

EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", "app/examples"))
SAMPLING_RATE = 100
DEVICE = torch.device("cpu")  # Spaces free tier is CPU-only

# --- Model + Grad-CAM (loaded once at startup) ---

def _resolve_checkpoint() -> Path | None:
    """Return a local path to the checkpoint, downloading from HF Hub if needed."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    if HF_MODEL_REPO:
        try:
            from huggingface_hub import hf_hub_download
            print(f"Fetching {HF_CKPT_FILENAME} from HF Hub repo {HF_MODEL_REPO}...")
            return Path(
                hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_CKPT_FILENAME)
            )
        except Exception as e:
            print(f"HF Hub download failed: {e}")
    return None


def load_model() -> tuple[torch.nn.Module, GradCAM1D]:
    model = resnet1d_medium(n_classes=5, n_leads=12)
    ckpt_path = _resolve_checkpoint()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(
            "WARNING: no checkpoint found locally or on HF Hub. "
            "Demo is running with a randomly initialised model."
        )
    model.to(DEVICE).eval()
    return model, GradCAM1D(model)


MODEL, CAM = load_model()


# --- Inference ---

def load_record(record_path: Path) -> np.ndarray:
    """Load a WFDB record and return (12, n_samples) float32 signal."""
    signal, _ = wfdb.rdsamp(str(record_path))  # (n_samples, 12)
    return signal.astype(np.float32)


def predict_and_explain(
    record_path: str | Path | None, target_class: str
) -> tuple[plt.Figure, str]:
    if not record_path:
        return _placeholder_figure("Pick or upload a record"), "No record selected."

    record_path = Path(record_path)
    # WFDB wants the path without extension
    if record_path.suffix in {".hea", ".dat"}:
        record_path = record_path.with_suffix("")

    try:
        raw = load_record(record_path)
    except Exception as e:
        return _placeholder_figure("Load failed"), f"Failed to read record: {e}"

    if raw.shape[1] != 12:
        return (
            _placeholder_figure("Wrong shape"),
            f"Expected 12 leads, got shape {raw.shape}.",
        )

    processed = preprocess_ecg(raw, fs=SAMPLING_RATE)        # (n_samples, 12)
    signal_t = torch.from_numpy(processed.T).float().unsqueeze(0).to(DEVICE)

    # Predictions
    with torch.no_grad():
        logits = MODEL(signal_t)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    prob_dict = {cls: float(p) for cls, p in zip(SUPERCLASSES, probs, strict=True)}

    # Grad-CAM
    target_idx = SUPERCLASS_TO_IDX[target_class]
    heatmap = CAM(signal_t, target_class=target_idx)

    fig = plot_prediction_summary(
        signal=processed.T,
        probabilities=prob_dict,
        heatmap=heatmap,
        target_class=target_class,
        sampling_rate=SAMPLING_RATE,
    )

    summary_lines = ["**Predicted probabilities:**", ""]
    for cls, p in sorted(prob_dict.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(round(p * 20))
        summary_lines.append(f"`{cls:>5}` {bar:<20} {p:.3f}")
    summary = "\n".join(summary_lines)

    return fig, summary


def _placeholder_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.axis("off")
    return fig


# --- Gradio UI ---

def gather_examples() -> list[list[str]]:
    """Find example records bundled with the app, if any."""
    if not EXAMPLES_DIR.exists():
        return []
    headers = sorted(EXAMPLES_DIR.glob("*.hea"))
    return [[str(h)] for h in headers]


DESCRIPTION = """
# ECG-Explain

A 12-lead ECG classifier that surfaces *why* it predicts what it predicts.
Per-lead Grad-CAM overlays highlight the waveform regions driving each diagnosis.

**How to use:** Upload a WFDB record (`.hea` + `.dat`), or pick one of the
bundled examples below. Choose a target class to explain, and the model will
show you which timepoints it focused on for that prediction.

Trained on PTB-XL (5 diagnostic superclasses). Built by a doctor who wanted
to trust the model before trusting the output.
""".strip()


with gr.Blocks(title="ECG-Explain") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            record_input = gr.File(
                label="Upload an ECG record (.hea file; .dat must be in the same folder)",
                file_types=[".hea"],
                type="filepath",
            )
            class_input = gr.Radio(
                SUPERCLASSES,
                value="MI",
                label="Class to explain",
                info="Grad-CAM will show which waveform regions drove this class's prediction.",
            )
            submit = gr.Button("Predict + Explain", variant="primary")
            summary_output = gr.Markdown(label="Predictions")

        with gr.Column(scale=2):
            fig_output = gr.Plot(label="ECG with Grad-CAM overlay")

    examples = gather_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[record_input],
            label="Example records (click to load)",
        )

    submit.click(
        fn=predict_and_explain,
        inputs=[record_input, class_input],
        outputs=[fig_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()