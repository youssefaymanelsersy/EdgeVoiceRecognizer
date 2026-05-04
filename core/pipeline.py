from __future__ import annotations

from typing import Sequence

import numpy as np

# Constants
SAMPLE_RATE = 8000
SAMPLE_LENGTH = 8000
FRAME_LENGTH = 250
FRAME_COUNT = 32
ADC_OFFSET = 512.0
ADC_MIN = 0.0
ADC_MAX = 1023.0
DEFAULT_NUM_MEL_FILTERS = 24


def preprocess_signal(raw_samples: Sequence[float] | np.ndarray) -> np.ndarray:
    signal = np.asarray(raw_samples, dtype=np.float32).reshape(-1)
    if signal.size != SAMPLE_LENGTH:
        raise ValueError(f"raw_samples must contain exactly {SAMPLE_LENGTH} samples")

    signal = np.clip(signal, ADC_MIN, ADC_MAX).astype(np.float32)
    signal = signal - ADC_OFFSET

    window_len = 100
    if window_len < 1:
        window_len = 1
    abs_signal = np.abs(signal)
    kernel = np.ones(window_len, dtype=np.float32) / float(window_len)
    energy = np.convolve(abs_signal, kernel, mode="same")

    peak = float(energy.max()) if energy.size else 0.0
    threshold = peak * 0.1
    if peak > 0.0:
        indices = np.nonzero(energy > threshold)[0]
        if indices.size > 0:
            start = int(indices[0])
            end = int(indices[-1])
            speech_center = (start + end) // 2
            desired_center = SAMPLE_LENGTH // 2
            shift = int(desired_center - speech_center)
            if shift != 0:
                shifted = np.zeros_like(signal)
                if shift > 0:
                    shifted[shift:] = signal[: SAMPLE_LENGTH - shift]
                else:
                    s = -shift
                    shifted[: SAMPLE_LENGTH - s] = signal[s:]
                signal = shifted

    abs_peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if abs_peak > 0.0:
        signal = signal / abs_peak

    pre_emphasis = 0.97
    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - pre_emphasis * signal[:-1]

    return emphasized.astype(np.float32, copy=False)


def extract_features(raw_samples: Sequence[float] | np.ndarray, n_mfcc: int = 10) -> np.ndarray:
    # Import locally to avoid circular imports at module import time
    from core.features import extract_features as _extract

    return _extract(raw_samples, n_mfcc=n_mfcc)
