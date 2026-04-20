"""Tests for YAML config loading."""

from __future__ import annotations

import textwrap

from ecg_explain.config import FullConfig


def test_load_config_from_yaml(tmp_path):
    yaml_text = textwrap.dedent("""
        data:
          data_dir: data/raw/ptbxl
          sampling_rate: 100
          apply_filter: true
          apply_normalisation: true

        model:
          name: resnet1d_small
          n_classes: 5
          n_leads: 12

        training:
          epochs: 2
          batch_size: 32
          lr: 0.001
          weight_decay: 0.0001
          early_stopping_patience: 5
          grad_clip: 1.0
          seed: 42
          device: cpu
          num_workers: 0
          checkpoint_dir: checkpoints/test
          use_class_weighting: false
    """).strip()

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_text)

    cfg = FullConfig.from_yaml(cfg_path)
    assert cfg.data["sampling_rate"] == 100
    assert cfg.model["name"] == "resnet1d_small"
    assert cfg.training["epochs"] == 2


def test_real_baseline_config_loads():
    """The actual config file in the repo should always be loadable."""
    cfg = FullConfig.from_yaml("configs/baseline.yaml")
    assert "data_dir" in cfg.data
    assert "name" in cfg.model
    assert cfg.training["epochs"] > 0


def test_real_smoke_config_loads():
    cfg = FullConfig.from_yaml("configs/smoke.yaml")
    assert cfg.training["epochs"] == 2
