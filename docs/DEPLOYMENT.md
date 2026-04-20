# Deploying to Hugging Face Spaces

The Gradio demo at [`app/app.py`](../app/app.py) is designed to deploy on a
free HF Spaces CPU instance. The trained model weights live separately on the
Hugging Face Hub (Spaces git is not for large binary files).

## One-time setup

You need a Hugging Face account: https://huggingface.co/join

Install the HF CLI and log in:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

(Paste a token from https://huggingface.co/settings/tokens with write access.)

## Step 1 — Upload the trained checkpoint to the Hub

After training, push the checkpoint to a model repo on the Hub:

```bash
huggingface-cli repo create ecg-explain-resnet1d --type model
huggingface-cli upload \
    M-Omarjee/ecg-explain-resnet1d \
    checkpoints/baseline/best.pt \
    best.pt
```

Replace `M-Omarjee` with your HF username if different.

## Step 2 — Create the Space

On the website: https://huggingface.co/new-space

- Name: `ecg-explain`
- License: MIT
- SDK: Gradio
- Hardware: CPU basic (free)

You'll get a git URL like `https://huggingface.co/spaces/M-Omarjee/ecg-explain`.

## Step 3 — Push the demo to the Space

Clone the (empty) Space repo somewhere outside the main repo:

```bash
cd ~/Desktop
git clone https://huggingface.co/spaces/M-Omarjee/ecg-explain ecg-explain-space
cd ecg-explain-space
```

Copy the demo files from your main repo:

```bash
cp -r ../Projects/ecg-explain/app/* .
cp -r ../Projects/ecg-explain/src .
cp ../Projects/ecg-explain/app/SPACE_README.md README.md
```

Open `README.md` in this Space repo and confirm the frontmatter at the top
is intact (the `---` block is what HF parses).

Set the env var that points the app at the Hub model:

```bash
echo "HF_MODEL_REPO=M-Omarjee/ecg-explain-resnet1d" > .env
```

(Or set it through the Space's Settings → Variables and secrets in the UI.)

Commit and push:

```bash
git add .
git commit -m "Deploy ECG-Explain demo"
git push
```

The Space will build (~3–5 minutes for first build) and the demo will be live
at `https://huggingface.co/spaces/M-Omarjee/ecg-explain`.

## Step 4 — Add the live link to the main README

Once the Space is live, add a badge to the top of your main `README.md`:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/M-Omarjee/ecg-explain)
```

## Updating the demo

Any time you push improvements to `app/app.py` or model weights:

- Code: `cp` to the Space repo, commit, push.
- Weights: `huggingface-cli upload` again with the new `.pt`. The Space
  re-downloads on next restart.