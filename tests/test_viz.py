"""Smoke tests for visualisation. Render figures to memory, no display needed."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for CI

import numpy as np
import pytest
from matplotlib.figure import Figure

from ecg_explain.viz import LEAD_NAMES, plot_12_lead, plot_prediction_summary


@pytest.fixture
def fake_ecg() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 1000)).astype(np.float32)


def test_lead_names_count():
    assert len(LEAD_NAMES) == 12


def test_plot_12_lead_returns_figure(fake_ecg):
    fig = plot_12_lead(fake_ecg)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 12


def test_plot_12_lead_with_heatmap(fake_ecg):
    heatmap = np.linspace(0, 1, 1000)
    fig = plot_12_lead(fake_ecg, heatmap=heatmap)
    assert isinstance(fig, Figure)


def test_plot_12_lead_rejects_wrong_shape():
    bad = np.zeros((6, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match="12 leads"):
        plot_12_lead(bad)


def test_plot_prediction_summary_runs(fake_ecg):
    probs = {"NORM": 0.1, "MI": 0.8, "STTC": 0.2, "CD": 0.05, "HYP": 0.1}
    heatmap = np.linspace(0, 1, 1000)
    fig = plot_prediction_summary(fake_ecg, probabilities=probs, heatmap=heatmap, target_class="MI")
    assert isinstance(fig, Figure)
