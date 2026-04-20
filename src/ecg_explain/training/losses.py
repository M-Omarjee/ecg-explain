"""Loss functions for multi-label ECG classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """Multi-label BCE with optional per-class positive weighting.

    PTB-XL is imbalanced (NORM ~44%, HYP ~12%). pos_weight upweights the
    positive class for each label independently, which improves recall on
    rare classes without hurting overall calibration.

    pos_weight should be computed from the training set as:
        pos_weight[c] = (n_negative_c / n_positive_c)
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def compute_pos_weight(labels: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Compute pos_weight from a (n_samples, n_classes) multi-hot label matrix.

    Args:
        labels: float or int tensor of shape (n_samples, n_classes)
        eps: floor on positive count to avoid div-by-zero
    """
    n_pos = labels.sum(dim=0).clamp(min=eps)
    n_neg = labels.shape[0] - n_pos
    return n_neg / n_pos