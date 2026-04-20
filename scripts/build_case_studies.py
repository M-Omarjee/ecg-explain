"""Generate a set of case-study figures for the README.

Picks 6 records from the test set:
    - One correctly-classified example per superclass (5 figures)
    - One high-confidence false positive or false negative (1 figure)

Output: figures/case_studies/*.png — drop these straight into the README.

Usage:
    uv run python scripts/build_case_studies.py \\
        --config configs/baseline.yaml \\
        --checkpoint checkpoints/baseline/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ecg_explain.config import FullConfig
from ecg_explain.data import PTBXLDataset
from ecg_explain.data.labels import SUPERCLASSES
from ecg_explain.interpret import GradCAM1D
from ecg_explain.training import get_device
from ecg_explain.viz import plot_prediction_summary
from scripts.train import build_model


@torch.no_grad()
def collect_all_predictions(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for signals, targets in tqdm(loader, desc="scoring test set"):
        logits = model(signals.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    return targets, 1 / (1 + np.exp(-logits))  # sigmoid


def pick_best_correct_per_class(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, int]:
    """For each class, find the most confident correct prediction (positive class)."""
    picks: dict[str, int] = {}
    for i, cls in enumerate(SUPERCLASSES):
        positive_mask = y_true[:, i] == 1
        if positive_mask.sum() == 0:
            continue
        candidate_scores = y_score[:, i].copy()
        candidate_scores[~positive_mask] = -np.inf
        picks[cls] = int(np.argmax(candidate_scores))
    return picks


def pick_high_confidence_failure(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> tuple[int, str, str] | None:
    """Find the most confident *wrong* prediction across the test set.

    Returns (record_idx, predicted_class, true_class_summary) or None.
    """
    y_pred = (y_score >= threshold).astype(int)
    # Per-record: did any class disagree?
    disagreements = (y_pred != y_true).any(axis=1)
    if not disagreements.any():
        return None

    # Score each disagreement by max wrong-class confidence
    wrong_class_score = np.where(y_pred != y_true, y_score, 0).max(axis=1)
    wrong_class_score[~disagreements] = -np.inf
    idx = int(np.argmax(wrong_class_score))

    pred_classes = [c for c, v in zip(SUPERCLASSES, y_pred[idx], strict=True) if v]
    true_classes = [c for c, v in zip(SUPERCLASSES, y_true[idx], strict=True) if v]
    pred_str = "+".join(pred_classes) or "none"
    true_str = "+".join(true_classes) or "none"
    return idx, pred_str, true_str


def render_one_case(
    idx: int,
    dataset: PTBXLDataset,
    model: torch.nn.Module,
    cam: GradCAM1D,
    target_class: str,
    title_prefix: str,
    out_path: Path,
    sampling_rate: int,
    device: torch.device,
) -> None:
    from ecg_explain.data.labels import SUPERCLASS_TO_IDX

    signal, labels = dataset[idx]
    signal_dev = signal.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(signal_dev))[0].cpu().numpy()
    prob_dict = {cls: float(p) for cls, p in zip(SUPERCLASSES, probs, strict=True)}

    heatmap = cam(signal_dev, target_class=SUPERCLASS_TO_IDX[target_class])

    fig = plot_prediction_summary(
        signal=signal.numpy(),
        probabilities=prob_dict,
        heatmap=heatmap,
        target_class=target_class,
        sampling_rate=sampling_rate,
    )
    if fig._suptitle:
        fig._suptitle.set_text(f"{title_prefix}\n{fig._suptitle.get_text()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}")


def main(config_path: str, checkpoint_path: str, out_dir: str) -> None:
    cfg = FullConfig.from_yaml(config_path)
    device = get_device(cfg.training["device"])

    test_ds = PTBXLDataset(
        data_dir=cfg.data["data_dir"],
        split="test",
        sampling_rate=cfg.data["sampling_rate"],
        apply_filter=cfg.data["apply_filter"],
        apply_normalisation=cfg.data["apply_normalisation"],
    )
    loader = DataLoader(
        test_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
    )

    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cam = GradCAM1D(model)

    print("Scoring test set to pick case studies...")
    y_true, y_score = collect_all_predictions(model, loader, device)

    out_dir = Path(out_dir)
    print("\nGenerating one correct example per class:")
    for cls, idx in pick_best_correct_per_class(y_true, y_score).items():
        render_one_case(
            idx=idx,
            dataset=test_ds,
            model=model,
            cam=cam,
            target_class=cls,
            title_prefix=f"Case study: correct {cls} (test idx {idx})",
            out_path=out_dir / f"correct_{cls}.png",
            sampling_rate=cfg.data["sampling_rate"],
            device=device,
        )

    print("\nGenerating one high-confidence failure:")
    failure = pick_high_confidence_failure(y_true, y_score)
    if failure is None:
        print("  no failures found (suspicious — model may be too confident)")
    else:
        idx, pred_str, true_str = failure
        # Explain the *predicted* class, since that's the one we want to interrogate
        first_pred = pred_str.split("+")[0]
        if first_pred not in SUPERCLASSES:
            first_pred = "MI"
        render_one_case(
            idx=idx,
            dataset=test_ds,
            model=model,
            cam=cam,
            target_class=first_pred,
            title_prefix=(
                f"Case study: failure (test idx {idx}) — "
                f"predicted [{pred_str}], true [{true_str}]"
            ),
            out_path=out_dir / "failure_high_confidence.png",
            sampling_rate=cfg.data["sampling_rate"],
            device=device,
        )

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="figures/case_studies")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.out_dir)