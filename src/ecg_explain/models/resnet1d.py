"""1D ResNet for 12-lead ECG classification.

Architecture inspired by the strongest baseline in Strodthoff et al. 2020
(PTB-XL benchmark), adapted with cleaner conventions:
    - Pre-activation residual blocks (BN -> ReLU -> Conv)
    - 4 stages with [2, 2, 2, 2] blocks (ResNet-18 style) by default
    - Configurable widths and depth via the factory functions

Input:  (batch, 12, n_samples)  — 12 leads, n_samples=1000 at 100Hz
Output: (batch, n_classes)       — raw logits (apply sigmoid for probabilities)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv1d(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> nn.Conv1d:
    """3-arg conv with 'same' padding for odd kernels."""
    padding = kernel_size // 2
    return nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)


class BasicBlock1D(nn.Module):
    """Pre-activation residual block.

    Pre-activation (BN -> ReLU -> Conv) tends to train more stably than the
    original post-activation ResNet design, especially for 1D signals.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = _conv1d(in_ch, out_ch, kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = _conv1d(out_ch, out_ch, kernel_size, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # Project shortcut if shape changes
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class ResNet1D(nn.Module):
    """1D ResNet for multi-label ECG classification."""

    def __init__(
        self,
        n_classes: int = 5,
        n_leads: int = 12,
        base_filters: int = 64,
        kernel_size: int = 7,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        stage_widths: tuple[int, ...] = (64, 128, 256, 512),
        stem_kernel_size: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(blocks_per_stage) == len(stage_widths), (
            "blocks_per_stage and stage_widths must have same length"
        )

        # Stem: large kernel to capture beat-scale patterns from raw signal
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_leads,
                base_filters,
                kernel_size=stem_kernel_size,
                stride=2,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        stages = []
        in_ch = base_filters
        for stage_idx, (n_blocks, width) in enumerate(
            zip(blocks_per_stage, stage_widths, strict=True)
        ):
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stages.append(
                    BasicBlock1D(in_ch, width, kernel_size=kernel_size, stride=stride)
                )
                in_ch = width
        self.stages = nn.Sequential(*stages)

        # Head
        self.head_norm = nn.BatchNorm1d(in_ch)
        self.head_relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_relu(self.head_norm(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final feature maps (before global pool).

        Used by Grad-CAM for attribution. Shape: (batch, channels, time).
        """
        x = self.stem(x)
        x = self.stages(x)
        return self.head_relu(self.head_norm(x))


# --- Factory functions for common sizes ---

def resnet1d_small(n_classes: int = 5, n_leads: int = 12) -> ResNet1D:
    """~1.5M params. Fast iteration."""
    return ResNet1D(
        n_classes=n_classes,
        n_leads=n_leads,
        blocks_per_stage=(1, 1, 1, 1),
        stage_widths=(32, 64, 128, 256),
    )


def resnet1d_medium(n_classes: int = 5, n_leads: int = 12) -> ResNet1D:
    """~6M params. Default for the headline result."""
    return ResNet1D(
        n_classes=n_classes,
        n_leads=n_leads,
        blocks_per_stage=(2, 2, 2, 2),
        stage_widths=(64, 128, 256, 512),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)