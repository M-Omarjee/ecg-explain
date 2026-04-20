"""Generate the README banner: synthetic ECG with Grad-CAM-style overlay.

Output: figures/banner.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def synthesise_ecg_lead(n_samples: int = 1000, fs: int = 100, seed: int = 0) -> np.ndarray:
    """Generate one lead of fake-but-plausible ECG via summed Gaussian bumps."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    signal = np.zeros_like(t)

    # ~60 bpm = 1 beat per second
    beat_period = 1.0
    n_beats = int(t[-1] / beat_period) + 1

    for beat in range(n_beats):
        beat_start = beat * beat_period

        # P wave (small bump before QRS)
        signal += 0.15 * np.exp(-((t - beat_start - 0.15) ** 2) / (2 * 0.02**2))
        # Q (small dip)
        signal -= 0.1 * np.exp(-((t - beat_start - 0.28) ** 2) / (2 * 0.005**2))
        # R (big spike)
        signal += 1.0 * np.exp(-((t - beat_start - 0.30) ** 2) / (2 * 0.008**2))
        # S (dip after R)
        signal -= 0.25 * np.exp(-((t - beat_start - 0.33) ** 2) / (2 * 0.008**2))
        # T wave (rounded bump after QRS)
        signal += 0.3 * np.exp(-((t - beat_start - 0.50) ** 2) / (2 * 0.04**2))

    # Mild noise + baseline wander
    signal += 0.02 * rng.standard_normal(n_samples)
    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    return signal


def synthesise_heatmap(n_samples: int = 1000, fs: int = 100) -> np.ndarray:
    """Heatmap that lights up around the ST segment of each beat."""
    t = np.arange(n_samples) / fs
    heatmap = np.zeros_like(t)
    for beat_start in np.arange(0, t[-1], 1.0):
        # ST segment is roughly 0.35-0.45s after beat start
        heatmap += np.exp(-((t - beat_start - 0.40) ** 2) / (2 * 0.05**2))
    # Normalise to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def make_banner(out_path: Path) -> None:
    n_samples, fs = 1000, 100
    t = np.arange(n_samples) / fs

    leads = [synthesise_ecg_lead(n_samples, fs, seed=i) for i in range(3)]
    heatmap = synthesise_heatmap(n_samples, fs)

    fig, axes = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
    fig.patch.set_facecolor("#0d1117")  # GitHub dark mode bg

    for ax, lead, name in zip(axes, leads, ["I", "II", "V5"], strict=True):
        # Heatmap as background
        ymin, ymax = lead.min() - 0.2, lead.max() + 0.2
        ax.imshow(
            heatmap[np.newaxis, :],
            aspect="auto",
            cmap="Reds",
            alpha=0.5,
            extent=(t[0], t[-1], ymin, ymax),
            zorder=-1,
        )
        ax.plot(t, lead, color="white", linewidth=1.2)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([])
        ax.set_facecolor("#0d1117")
        ax.text(
            -0.02,
            0.5,
            name,
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("Time (s)", color="white")
    axes[-1].tick_params(axis="x", colors="white")

    fig.suptitle(
        "ECG-Explain — 12-lead classifier with per-lead Grad-CAM",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved banner to {out_path}")


if __name__ == "__main__":
    make_banner(Path("figures/banner.png"))
