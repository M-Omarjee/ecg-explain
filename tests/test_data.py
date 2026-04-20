"""Smoke tests for the data module.

Tests that don't need the full PTB-XL download. Dataset integration tests
are gated behind a skipif so CI passes without the dataset present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ecg_explain.data.labels import (
    SUPERCLASS_TO_IDX,
    SUPERCLASSES,
    parse_scp_codes,
    scp_to_superclass_labels,
)
from ecg_explain.data.preprocessing import (
    bandpass_filter,
    preprocess_ecg,
    z_normalise,
)


def test_superclasses_count():
    assert len(SUPERCLASSES) == 5
    assert set(SUPERCLASSES) == {"NORM", "MI", "STTC", "CD", "HYP"}


def test_superclass_idx_consistent():
    for cls, idx in SUPERCLASS_TO_IDX.items():
        assert SUPERCLASSES[idx] == cls


def test_parse_scp_codes():
    parsed = parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}")
    assert parsed == {"NORM": 100.0, "SR": 0.0}


def test_scp_to_superclass_labels_normal():
    mapping = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI"}
    labels = scp_to_superclass_labels("{'NORM': 100.0}", mapping)
    expected = np.zeros(5, dtype=np.float32)
    expected[SUPERCLASS_TO_IDX["NORM"]] = 1.0
    np.testing.assert_array_equal(labels, expected)


def test_scp_to_superclass_labels_multilabel():
    mapping = {"IMI": "MI", "NDT": "STTC"}
    labels = scp_to_superclass_labels("{'IMI': 100.0, 'NDT': 50.0}", mapping)
    expected = np.zeros(5, dtype=np.float32)
    expected[SUPERCLASS_TO_IDX["MI"]] = 1.0
    expected[SUPERCLASS_TO_IDX["STTC"]] = 1.0
    np.testing.assert_array_equal(labels, expected)


def test_bandpass_filter_shape():
    signal = np.random.randn(1000, 12).astype(np.float32)
    assert bandpass_filter(signal, fs=100).shape == signal.shape


def test_z_normalise_zero_mean_unit_std():
    signal = np.random.randn(1000, 12).astype(np.float32) * 5 + 10
    out = z_normalise(signal)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-3)


def test_preprocess_ecg_dtype_and_shape():
    signal = np.random.randn(1000, 12)
    out = preprocess_ecg(signal, fs=100)
    assert out.dtype == np.float32
    assert out.shape == (1000, 12)


# --- Integration test (only runs if data is downloaded) ---

DATA_PATH = Path("data/raw/ptbxl/ptbxl_database.csv")


@pytest.mark.skipif(not DATA_PATH.exists(), reason="PTB-XL not downloaded yet")
def test_dataset_loads_a_record():
    from ecg_explain.data.dataset import PTBXLDataset

    ds = PTBXLDataset("data/raw/ptbxl", split="val")
    assert len(ds) > 0
    signal, label = ds[0]
    assert signal.shape == (12, 1000)
    assert label.shape == (5,)
    assert label.sum() > 0  # has at least one class
