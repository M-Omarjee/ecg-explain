"""Generate a Grad-CAM overlay for a single record from a trained checkpoint.

Usage:
    uv run python scripts/explain.py \\
        --config configs/baseline.yaml \\
        --checkpoint checkpoints/baseline/best.pt \\
        --record-idx 0 \\
        --target-class MI \\
        --output figures/explain_record0_MI.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecg_explain.config import FullConfig
from ecg_explain.data import PTBXLDataset
from ecg_explain.data.labels import SUPERCLASS_TO_IDX, SUPERCLASSES
from ecg_explain.interpret import GradCAM1D
from ecg_explain.training import get_device
from ecg_explain.viz import plot_prediction_summary
from scripts.train import build_model


def main(
    config_path: str,
    checkpoint_path: str,
    record_idx: int,
    target_class: str,
    split: str,
    output_path: str,
) -> None:
    if target_class not in SUPERCLASS_TO_IDX:
        raise ValueError(f"target_class must be one of {SUPERCLASSES}, got {target_class!r}")
    target_idx = SUPERCLASS_TO_IDX[target_class]

    cfg = FullConfig.from_yaml(config_path)
    device = get_device(cfg.training["device"])

    # Data
    ds = PTBXLDataset(
        data_dir=cfg.data["data_dir"],
        split=split,
        sampling_rate=cfg.data["sampling_rate"],
        apply_filter=cfg.data["apply_filter"],
        apply_normalisation=cfg.data["apply_normalisation"],
    )
    signal, labels = ds[record_idx]
    print(f"Record idx {record_idx} ({split}) — true labels: "
          f"{[c for c, v in zip(SUPERCLASSES, labels.tolist(), strict=True) if v > 0]}")

    # Model + checkpoint
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Predictions
    signal_dev = signal.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(signal_dev)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    prob_dict = {cls: float(p) for cls, p in zip(SUPERCLASSES, probs, strict=True)}

    # Grad-CAM
    cam = GradCAM1D(model)
    heatmap = cam(signal_dev, target_class=target_idx)

    # Plot + save
    fig = plot_prediction_summary(
        signal=signal.numpy(),
        probabilities=prob_dict,
        heatmap=heatmap,
        target_class=target_class,
        sampling_rate=cfg.data["sampling_rate"],
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    print(f"Predicted probabilities: {prob_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--record-idx", type=int, default=0)
    parser.add_argument(
        "--target-class",
        default="MI",
        choices=SUPERCLASSES,
        help="Class to explain (NORM, MI, STTC, CD, HYP)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="figures/explanation.png")
    args = parser.parse_args()
    main(
        args.config,
        args.checkpoint,
        args.record_idx,
        args.target_class,
        args.split,
        args.output,
    )