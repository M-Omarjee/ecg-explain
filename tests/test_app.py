"""Smoke test for the Gradio app — verifies it imports and key callables run."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def test_app_imports():
    """Importing the app module shouldn't raise (loads model with random weights
    if checkpoint is absent)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    app = importlib.import_module("app.app")
    assert hasattr(app, "demo")
    assert hasattr(app, "predict_and_explain")


def test_placeholder_figure_renders():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    app = importlib.import_module("app.app")
    fig = app._placeholder_figure("hello")
    assert fig is not None