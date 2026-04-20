"""Training loop for ECG classification.

Plain PyTorch — no Lightning, no Hydra. Apple Silicon (MPS) supported.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ecg_explain.training.metrics import macro_auroc


def get_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device. 'auto' tries MPS -> CUDA -> CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0  # macOS + MPS — keep at 0 to avoid fork issues
    checkpoint_dir: str = "checkpoints"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_auroc: float
    lr: float
    epoch_time_s: float


@dataclass
class TrainHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = -1
    best_val_auroc: float = -float("inf")

    def to_dict(self) -> dict:
        return {
            "epochs": [asdict(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_auroc": self.best_val_auroc,
        }


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """End-to-end training loop with checkpointing and early stopping.

    Usage:
        trainer = Trainer(model, loss_fn, train_loader, val_loader, config)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ):
        self.config = config
        set_seed(config.seed)

        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = TrainHistory()

    # ---- Train / eval steps ----

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for signals, targets in pbar:
            signals = signals.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(signals)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for signals, targets in tqdm(self.val_loader, desc="val", leave=False):
            signals = signals.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(signals)
            loss = self.loss_fn(logits, targets)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        logits = torch.cat(all_logits).numpy()
        targets = torch.cat(all_targets).numpy()
        scores = 1 / (1 + np.exp(-logits))  # sigmoid

        return total_loss / max(n_batches, 1), macro_auroc(targets, scores)

    # ---- Checkpointing ----

    def _save_checkpoint(self, name: str, epoch: int, val_auroc: float) -> None:
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_macro_auroc": val_auroc,
                "config": asdict(self.config),
            },
            path,
        )

    # ---- Public API ----

    def fit(self) -> TrainHistory:
        print(f"Training on device: {self.device}")
        epochs_since_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss, val_auroc = self._validate()
            self.scheduler.step()
            elapsed = time.time() - t0

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_macro_auroc=val_auroc,
                lr=self.optimizer.param_groups[0]["lr"],
                epoch_time_s=elapsed,
            )
            self.history.epochs.append(metrics)
            print(
                f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_AUROC {val_auroc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_auroc > self.history.best_val_auroc:
                self.history.best_val_auroc = val_auroc
                self.history.best_epoch = epoch
                self._save_checkpoint("best", epoch, val_auroc)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.config.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} (no improvement for "
                        f"{self.config.early_stopping_patience} epochs)"
                    )
                    break

        self._save_checkpoint("last", epoch, val_auroc)
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)

        print(
            f"\nBest val AUROC: {self.history.best_val_auroc:.4f} "
            f"at epoch {self.history.best_epoch}"
        )
        return self.history
