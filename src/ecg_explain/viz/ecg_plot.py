"""Clinical-style 12-lead ECG plotting with optional Grad-CAM overlays."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Standard 12-lead ordering used in PTB-XL
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_12_lead(
    signal: np.ndarray,
    sampling_rate: int = 100,
    heatmap: np.ndarray | None = None,
    title: str | None = None,
    cmap: str = "Reds",
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Plot a 12-lead ECG in a 6x2 grid with optional Grad-CAM overlay.

    Args:
        signal: shape (12, n_samples) — channels first
        sampling_rate: Hz, used for the time axis
        heatmap: shape (n_samples,) attribution scores in [0, 1], or None.
                 If provided, drawn as a coloured background behind each lead.
        title: figure suptitle
        cmap: matplotlib colourmap for the heatmap
    """
    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got shape {signal.shape}")

    n_samples = signal.shape[1]
    t = np.arange(n_samples) / sampling_rate

    fig, axes = plt.subplots(6, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    # Order leads down columns: I, II, III, aVR, aVL, aVF | V1..V6
    col_order = [
        [0, 6],  # I,    V1
        [1, 7],  # II,   V2
        [2, 8],  # III,  V3
        [3, 9],  # aVR,  V4
        [4, 10],  # aVL,  V5
        [5, 11],  # aVF,  V6
    ]

    for row in range(6):
        for col in range(2):
            lead_idx = col_order[row][col]
            ax = axes[row, col]
            ax.plot(t, signal[lead_idx], color="black", linewidth=0.8)
            ax.set_ylabel(
                LEAD_NAMES[lead_idx],
                rotation=0,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.grid(True, which="both", color="lightgray", linewidth=0.4)
            ax.set_yticks([])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            if heatmap is not None:
                # Background colour band whose intensity follows the heatmap
                _overlay_heatmap(ax, t, heatmap, cmap=cmap)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    return fig


def _overlay_heatmap(ax, t: np.ndarray, heatmap: np.ndarray, cmap: str) -> None:
    """Render the heatmap as a coloured background band on the given axis."""
    # imshow lets us paint a continuous gradient as the axis background
    ymin, ymax = ax.get_ylim()
    ax.imshow(
        heatmap[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        alpha=0.35,
        extent=(t[0], t[-1], ymin, ymax),
        zorder=-1,
    )
    ax.set_ylim(ymin, ymax)


def plot_prediction_summary(
    signal: np.ndarray,
    probabilities: dict[str, float],
    heatmap: np.ndarray | None = None,
    target_class: str | None = None,
    sampling_rate: int = 100,
) -> Figure:
    """Convenience wrapper: ECG + Grad-CAM with a title summarising predictions."""
    pred_str = " | ".join(f"{cls}={p:.2f}" for cls, p in probabilities.items())
    title = f"Predictions: {pred_str}"
    if target_class:
        title = f"Grad-CAM for {target_class}\n{title}"
    return plot_12_lead(signal, sampling_rate=sampling_rate, heatmap=heatmap, title=title)
