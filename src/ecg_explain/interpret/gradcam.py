"""1D Grad-CAM for ECG classifiers.

Produces a (n_samples,) attribution map for a chosen class, indicating which
timepoints in the input signal most influenced that class's prediction.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization." ICCV 2017. (1D adaptation here.)

Usage:
    cam = GradCAM1D(model)
    heatmap = cam(signal_tensor, target_class=1)  # MI
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM1D:
    """Class activation mapping for 1D conv models.

    Assumes the model exposes a `feature_maps(x)` method returning the
    pre-pool conv features as (batch, channels, time). All ResNet1D
    instances in this repo do.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        if not hasattr(model, "feature_maps"):
            raise AttributeError(
                "Model must expose a `feature_maps(x)` method "
                "(see ecg_explain.models.ResNet1D)."
            )

    def __call__(
        self,
        signal: torch.Tensor,
        target_class: int,
        upsample_to: int | None = None,
        normalise: bool = True,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for one signal and one class.

        Args:
            signal: (12, n_samples) or (1, 12, n_samples)
            target_class: integer class index (0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP)
            upsample_to: length to upsample the heatmap to. Defaults to
                the input signal length.
            normalise: if True, scale the heatmap to [0, 1] for plotting.

        Returns:
            np.ndarray of shape (upsample_to,)
        """
        was_training = self.model.training
        self.model.eval()

        if signal.dim() == 2:
            signal = signal.unsqueeze(0)
        signal = signal.clone().requires_grad_(False)
        n_samples = signal.shape[-1]
        target_len = upsample_to or n_samples

        # Forward pass through stem + stages, then through the head separately
        # so we can grab feature maps with grad enabled.
        feats = self.model.feature_maps(signal)            # (1, C, T')
        feats.retain_grad()

        pooled = F.adaptive_avg_pool1d(feats, 1).squeeze(-1)   # (1, C)
        pooled = self.model.dropout(pooled)
        logits = self.model.classifier(pooled)             # (1, n_classes)

        # Backward pass for the chosen class only
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Channel importance = global-avg-pooled gradient
        grads = feats.grad                                 # (1, C, T')
        weights = grads.mean(dim=2, keepdim=True)          # (1, C, 1)

        # Weighted sum across channels, then ReLU
        cam = (weights * feats).sum(dim=1, keepdim=True)   # (1, 1, T')
        cam = F.relu(cam)

        # Upsample back to signal length
        cam = F.interpolate(cam, size=target_len, mode="linear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        if normalise:
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

        if was_training:
            self.model.train()
        return cam