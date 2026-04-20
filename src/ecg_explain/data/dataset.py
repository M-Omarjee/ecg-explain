"""PyTorch Dataset for PTB-XL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from ecg_explain.data.labels import (
    SUPERCLASSES,
    load_scp_mapping,
    scp_to_superclass_labels,
)
from ecg_explain.data.preprocessing import preprocess_ecg


class PTBXLDataset(Dataset):
    """PTB-XL multi-label classification dataset.

    Each item is a tuple (signal, labels) where:
        signal: torch.FloatTensor of shape (12, n_samples) — 12 leads
                (n_samples = 1000 at 100Hz, 5000 at 500Hz)
        labels: torch.FloatTensor of shape (5,) — multi-hot for
                [NORM, MI, STTC, CD, HYP]

    Uses the official PTB-XL stratified folds:
        - train: folds 1-8
        - val:   fold 9
        - test:  fold 10
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        sampling_rate: int = 100,
        apply_filter: bool = True,
        apply_normalisation: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.apply_filter = apply_filter
        self.apply_normalisation = apply_normalisation

        metadata = pd.read_csv(self.data_dir / "ptbxl_database.csv", index_col="ecg_id")
        scp_mapping = load_scp_mapping(self.data_dir / "scp_statements.csv")

        if split == "train":
            metadata = metadata[metadata.strat_fold <= 8]
        elif split == "val":
            metadata = metadata[metadata.strat_fold == 9]
        elif split == "test":
            metadata = metadata[metadata.strat_fold == 10]
        else:
            raise ValueError(f"Unknown split: {split!r}")

        labels = np.stack([scp_to_superclass_labels(s, scp_mapping) for s in metadata.scp_codes])

        # Drop records with no superclass label assigned
        keep = labels.sum(axis=1) > 0
        metadata = metadata[keep]
        labels = labels[keep]

        filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

        self.records = metadata[filename_col].values
        self.labels = labels
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_path = self.data_dir / self.records[idx]
        signal, _ = wfdb.rdsamp(str(record_path))  # (n_samples, 12)
        signal = preprocess_ecg(
            signal,
            fs=self.sampling_rate,
            apply_filter=self.apply_filter,
            apply_normalisation=self.apply_normalisation,
        )
        # PyTorch conv1d expects (channels, length), so transpose
        signal = signal.T
        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )

    def class_counts(self) -> dict[str, int]:
        """Useful for sanity checks and class-weighting later."""
        return {cls: int(self.labels[:, i].sum()) for i, cls in enumerate(SUPERCLASSES)}
