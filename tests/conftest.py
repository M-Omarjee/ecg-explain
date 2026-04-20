"""Shared pytest config — adds repo root to sys.path so tests can import
top-level packages like `app` and `scripts` without per-test boilerplate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
