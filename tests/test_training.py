"""Tests for losses, metrics, and trainer plumbing.

No real data needed — uses synthetic tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ecg_explain.models import resnet1d_small
from ecg_explain.training import (
    TrainConfig,
    Trainer,
    WeightedBCEWithLogitsLoss,
    compute_all_metrics,
    compute_pos_weight,
    get_device,
    macro_auroc,
    macro_f1,
    per_class_auroc,
    per_class_f1,
    set_seed,
)
from torch.utils.data import DataLoader, TensorDataset

# --- Losses ---


def test_bce_loss_runs():
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_bce_loss_with_pos_weight():
    pos_weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_compute_pos_weight():
    # 100 samples, class 0 always positive, class 1 never positive
    labels = torch.zeros(100, 2)
    labels[:, 0] = 1.0
    pw = compute_pos_weight(labels)
    assert pw[0] < 1.0  # class 0 is over-represented, downweighted
    assert pw[1] > 1.0  # class 1 is rare, upweighted


# --- Metrics ---


def _fake_predictions(n_samples: int = 200, n_classes: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n_samples, n_classes))
    # Bias scores toward truth so AUROC > 0.5
    y_score = y_true * rng.uniform(0.5, 1.0, size=y_true.shape) + (1 - y_true) * rng.uniform(
        0.0, 0.5, size=y_true.shape
    )
    return y_true, y_score


def test_per_class_auroc_keys():
    y_true, y_score = _fake_predictions()
    out = per_class_auroc(y_true, y_score)
    assert set(out.keys()) == {"NORM", "MI", "STTC", "CD", "HYP"}


def test_macro_auroc_above_chance():
    y_true, y_score = _fake_predictions()
    score = macro_auroc(y_true, y_score)
    assert score > 0.5


def test_per_class_f1_returns_floats():
    y_true, y_score = _fake_predictions()
    out = per_class_f1(y_true, y_score)
    assert all(isinstance(v, float) for v in out.values())


def test_macro_f1_in_unit_interval():
    y_true, y_score = _fake_predictions()
    f1 = macro_f1(y_true, y_score)
    assert 0.0 <= f1 <= 1.0


def test_compute_all_metrics_structure():
    y_true, y_score = _fake_predictions()
    out = compute_all_metrics(y_true, y_score)
    assert set(out.keys()) == {"macro_auroc", "per_class_auroc", "macro_f1", "per_class_f1"}


def test_per_class_auroc_handles_empty_class():
    y_true = np.zeros((50, 5), dtype=int)
    y_true[:, 0] = np.random.randint(0, 2, 50)  # only class 0 has positives
    y_score = np.random.rand(50, 5)
    out = per_class_auroc(y_true, y_score)
    assert not np.isnan(out["NORM"])
    assert all(np.isnan(out[c]) for c in ["MI", "STTC", "CD", "HYP"])


# --- Device + seed ---


def test_get_device_cpu():
    assert get_device("cpu").type == "cpu"


def test_get_device_auto_returns_valid():
    dev = get_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


# --- Trainer end-to-end on tiny synthetic data ---


@pytest.fixture
def tiny_loaders():
    """16 fake ECGs, 4 fake ECGs for val. CPU only."""
    torch.manual_seed(0)
    train_x = torch.randn(16, 12, 1000)
    train_y = torch.randint(0, 2, (16, 5)).float()
    val_x = torch.randn(4, 12, 1000)
    val_y = torch.randint(0, 2, (4, 5)).float()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=4)
    return train_loader, val_loader


def test_trainer_runs_two_epochs(tmp_path, tiny_loaders):
    train_loader, val_loader = tiny_loaders
    model = resnet1d_small()
    loss_fn = WeightedBCEWithLogitsLoss()
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        early_stopping_patience=10,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    trainer = Trainer(model, loss_fn, train_loader, val_loader, config)
    history = trainer.fit()

    assert len(history.epochs) == 2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "history.json").exists()


def test_trainer_checkpoint_loads_back(tmp_path, tiny_loaders):
    train_loader, val_loader = tiny_loaders
    model = resnet1d_small()
    loss_fn = WeightedBCEWithLogitsLoss()
    config = TrainConfig(epochs=1, device="cpu", checkpoint_dir=str(tmp_path))
    Trainer(model, loss_fn, train_loader, val_loader, config).fit()

    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    assert "model_state" in ckpt
    assert "config" in ckpt
    new_model = resnet1d_small()
    new_model.load_state_dict(ckpt["model_state"])
