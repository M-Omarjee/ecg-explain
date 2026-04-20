"""Tests for Grad-CAM. Uses an untrained model — we're checking shape and
behaviour, not attribution quality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_explain.interpret import GradCAM1D
from ecg_explain.models import resnet1d_small


@pytest.fixture
def model_and_signal():
    torch.manual_seed(0)
    model = resnet1d_small()
    signal = torch.randn(12, 1000)
    return model, signal


def test_gradcam_returns_correct_length(model_and_signal):
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    heatmap = cam(signal, target_class=1)
    assert heatmap.shape == (1000,)


def test_gradcam_handles_batch_input(model_and_signal):
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    batched = signal.unsqueeze(0)  # (1, 12, 1000)
    heatmap = cam(batched, target_class=0)
    assert heatmap.shape == (1000,)


def test_gradcam_normalised_in_unit_interval(model_and_signal):
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    heatmap = cam(signal, target_class=2, normalise=True)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_unnormalised_is_nonnegative(model_and_signal):
    """Grad-CAM applies ReLU, so unnormalised values must be >= 0."""
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    heatmap = cam(signal, target_class=2, normalise=False)
    assert heatmap.min() >= 0.0


def test_gradcam_upsample_target_length(model_and_signal):
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    heatmap = cam(signal, target_class=0, upsample_to=2000)
    assert heatmap.shape == (2000,)


def test_gradcam_different_classes_give_different_heatmaps(model_and_signal):
    model, signal = model_and_signal
    cam = GradCAM1D(model)
    h0 = cam(signal, target_class=0)
    h1 = cam(signal, target_class=1)
    assert not np.allclose(h0, h1)


def test_gradcam_rejects_model_without_feature_maps():
    bad_model = torch.nn.Linear(10, 5)
    with pytest.raises(AttributeError, match="feature_maps"):
        GradCAM1D(bad_model)


def test_gradcam_does_not_change_model_mode(model_and_signal):
    model, signal = model_and_signal
    model.train()
    cam = GradCAM1D(model)
    cam(signal, target_class=0)
    assert model.training is True

    model.eval()
    cam(signal, target_class=0)
    assert model.training is False
