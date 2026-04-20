"""Signal preprocessing for ECG."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: int = 100,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    0.5-40Hz removes baseline wander (low) and EMG/mains noise (high)
    while preserving the diagnostic morphology (P, QRS, T).

    Args:
        signal: shape (n_samples,) or (n_samples, n_leads)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def z_normalise(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalise each lead independently."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return (signal - mean) / (std + eps)


def preprocess_ecg(
    signal: np.ndarray,
    fs: int = 100,
    apply_filter: bool = True,
    apply_normalisation: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline. Returns float32."""
    if apply_filter:
        signal = bandpass_filter(signal, fs=fs)
    if apply_normalisation:
        signal = z_normalise(signal)
    return signal.astype(np.float32)