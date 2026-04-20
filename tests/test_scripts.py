"""Smoke tests: importable scripts shouldn't blow up at import time."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _add_repo_root_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_train_script_imports():
    _add_repo_root_to_path()
    importlib.import_module("scripts.train")


def test_evaluate_script_imports():
    _add_repo_root_to_path()
    importlib.import_module("scripts.evaluate")


def test_explain_script_imports():
    _add_repo_root_to_path()
    importlib.import_module("scripts.explain")


def test_build_case_studies_script_imports():
    _add_repo_root_to_path()
    importlib.import_module("scripts.build_case_studies")