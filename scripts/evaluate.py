"""Evaluate a trained checkpoint on the test set.

Usage:
    uv run python scripts/evaluate.py \\
        --config configs/baseline.yaml \\
        --checkpoint checkpoints/baseline/best.pt \\
        --output results/baseline_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ecg_explain.config import FullConfig
from ecg_explain.data import PTBXLDataset
from ecg_explain.training import compute_all_metrics, get_device
from scripts.train import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for signals, targets in tqdm(loader, desc="eval"):
        signals = signals.to(device)
        logits = model(signals)
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    scores = 1 / (1 + np.exp(-logits))  # sigmoid
    return targets, scores


def main(config_path: str, checkpoint_path: str, output_path: str) -> None:
    cfg = FullConfig.from_yaml(config_path)
    device = get_device(cfg.training["device"])
    print(f"Using device: {device}")

    # Data
    test_ds = PTBXLDataset(
        data_dir=cfg.data["data_dir"],
        split="test",
        sampling_rate=cfg.data["sampling_rate"],
        apply_filter=cfg.data["apply_filter"],
        apply_normalisation=cfg.data["apply_normalisation"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
    )
    print(f"Test set: {len(test_ds)} records")

    # Model + checkpoint
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(
        f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt['epoch']}, "
        f"val_AUROC={ckpt['val_macro_auroc']:.4f})"
    )

    # Eval
    y_true, y_score = collect_predictions(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_score)

    # Pretty print
    print("\n=== Test set metrics ===")
    print(f"  Macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print("  Per-class AUROC:")
    for cls, score in metrics["per_class_auroc"].items():
        print(f"    {cls:>5}: {score:.4f}")
    print("  Per-class F1:")
    for cls, score in metrics["per_class_f1"].items():
        print(f"    {cls:>5}: {score:.4f}")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "config": cfg.to_dict(),
                "metrics": metrics,
                "n_test_samples": len(test_ds),
            },
            f,
            indent=2,
        )
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG classifier.")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file path")
    parser.add_argument(
        "--output",
        default="results/test_metrics.json",
        help="Where to save the metrics JSON",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output)
