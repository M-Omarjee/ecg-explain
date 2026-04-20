"""Train an ECG classifier from a YAML config.

Usage:
    uv run python scripts/train.py configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ecg_explain.config import FullConfig
from ecg_explain.data import PTBXLDataset
from ecg_explain.models import resnet1d_medium, resnet1d_small
from ecg_explain.training import (
    TrainConfig,
    Trainer,
    WeightedBCEWithLogitsLoss,
    compute_pos_weight,
)

MODEL_REGISTRY = {
    "resnet1d_small": resnet1d_small,
    "resnet1d_medium": resnet1d_medium,
}


def build_model(model_cfg: dict) -> torch.nn.Module:
    name = model_cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        n_classes=model_cfg["n_classes"],
        n_leads=model_cfg["n_leads"],
    )


def main(config_path: str) -> None:
    cfg = FullConfig.from_yaml(config_path)

    print(f"Loaded config from {config_path}")
    print(json.dumps(cfg.to_dict(), indent=2))

    # --- Data ---
    print("\nLoading datasets...")
    train_ds = PTBXLDataset(
        data_dir=cfg.data["data_dir"],
        split="train",
        sampling_rate=cfg.data["sampling_rate"],
        apply_filter=cfg.data["apply_filter"],
        apply_normalisation=cfg.data["apply_normalisation"],
    )
    val_ds = PTBXLDataset(
        data_dir=cfg.data["data_dir"],
        split="val",
        sampling_rate=cfg.data["sampling_rate"],
        apply_filter=cfg.data["apply_filter"],
        apply_normalisation=cfg.data["apply_normalisation"],
    )
    print(f"  Train: {len(train_ds):>6} records — {train_ds.class_counts()}")
    print(f"  Val:   {len(val_ds):>6} records — {val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=cfg.training["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
        pin_memory=False,
    )

    # --- Model + loss ---
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cfg.model['name']} — {n_params:,} parameters")

    if cfg.training.get("use_class_weighting", False):
        labels_tensor = torch.from_numpy(train_ds.labels)
        pos_weight = compute_pos_weight(labels_tensor)
        print(f"Class pos_weights: {pos_weight.tolist()}")
    else:
        pos_weight = None

    loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training ---
    train_cfg = TrainConfig(
        epochs=cfg.training["epochs"],
        batch_size=cfg.training["batch_size"],
        lr=cfg.training["lr"],
        weight_decay=cfg.training["weight_decay"],
        early_stopping_patience=cfg.training["early_stopping_patience"],
        grad_clip=cfg.training["grad_clip"],
        seed=cfg.training["seed"],
        device=cfg.training["device"],
        num_workers=cfg.training["num_workers"],
        checkpoint_dir=cfg.training["checkpoint_dir"],
    )

    trainer = Trainer(model, loss_fn, train_loader, val_loader, train_cfg)
    history = trainer.fit()

    # Dump the full config alongside checkpoints for reproducibility
    out_dir = Path(cfg.training["checkpoint_dir"])
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(f"\nDone. Best val macro AUROC: {history.best_val_auroc:.4f}")
    print(f"Checkpoints saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ECG classifier.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    main(args.config)