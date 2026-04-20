"""Config loading from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FullConfig:
    """Top-level config, loaded from YAML."""

    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> FullConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            data=raw["data"],
            model=raw["model"],
            training=raw["training"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "model": self.model, "training": self.training}
