"""Model architecture tests. No data needed — uses random tensors."""
from __future__ import annotations

import pytest
import torch

from ecg_explain.models import (
    BasicBlock1D,
    ResNet1D,
    count_parameters,
    resnet1d_medium,
    resnet1d_small,
)

# --- Fixtures ---

@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """A batch of 4 random 'ECGs': 12 leads, 1000 samples (10s at 100Hz)."""
    return torch.randn(4, 12, 1000)


# --- BasicBlock1D ---

def test_basic_block_preserves_shape_when_no_stride():
    block = BasicBlock1D(in_ch=32, out_ch=32, stride=1)
    x = torch.randn(2, 32, 100)
    out = block(x)
    assert out.shape == (2, 32, 100)


def test_basic_block_downsamples_with_stride():
    block = BasicBlock1D(in_ch=32, out_ch=64, stride=2)
    x = torch.randn(2, 32, 100)
    out = block(x)
    assert out.shape == (2, 64, 50)


def test_basic_block_changes_channels():
    block = BasicBlock1D(in_ch=16, out_ch=64, stride=1)
    x = torch.randn(2, 16, 100)
    out = block(x)
    assert out.shape == (2, 64, 100)


# --- ResNet1D forward pass ---

def test_resnet_small_forward_shape(dummy_batch):
    model = resnet1d_small()
    out = model(dummy_batch)
    assert out.shape == (4, 5)


def test_resnet_medium_forward_shape(dummy_batch):
    model = resnet1d_medium()
    out = model(dummy_batch)
    assert out.shape == (4, 5)


def test_resnet_custom_classes(dummy_batch):
    model = ResNet1D(n_classes=10)
    out = model(dummy_batch)
    assert out.shape == (4, 10)


def test_resnet_outputs_are_logits(dummy_batch):
    """Logits, not probabilities — should not be bounded to [0, 1]."""
    model = resnet1d_medium()
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch)
    assert out.min() < 0 or out.max() > 1, "Output looks like probabilities, expected logits"


# --- Feature maps for Grad-CAM ---

def test_feature_maps_shape(dummy_batch):
    """Grad-CAM needs (batch, channels, time) feature maps."""
    model = resnet1d_medium()
    feats = model.feature_maps(dummy_batch)
    assert feats.dim() == 3
    assert feats.shape[0] == 4
    assert feats.shape[1] == 512  # final stage width for medium


# --- Backward pass (gradients flow) ---

def test_backward_pass_runs(dummy_batch):
    model = resnet1d_medium()
    targets = torch.randint(0, 2, (4, 5)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(dummy_batch), targets)
    loss.backward()
    # Check at least one gradient is non-zero
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)


# --- Param count sanity ---

def test_small_smaller_than_medium():
    assert count_parameters(resnet1d_small()) < count_parameters(resnet1d_medium())


def test_medium_param_count_in_expected_range():
    n = count_parameters(resnet1d_medium())
    # Should be in the low millions
    assert 3_000_000 < n < 15_000_000, f"Unexpected param count: {n:,}"