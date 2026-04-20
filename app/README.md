# Gradio demo

Browser-based UI for the trained ECG classifier.

## Run locally

From the repo root:

    uv sync --all-extras
    uv run python app/app.py

Then open the URL shown in the terminal (usually http://127.0.0.1:7860).

## Examples

Drop WFDB record pairs (`.hea` + `.dat`) into `app/examples/`. Each `.hea`
file will appear as a clickable example in the UI.

## Deployment

This app is also deployed as a Hugging Face Space. The `Spacefile`-equivalent
configuration lives at the top of `app/app.py` and reads the checkpoint path
from the `CKPT_PATH` environment variable.