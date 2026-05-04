from __future__ import annotations

from typing import Sequence

import numpy as np

from core.preprocess import preprocess_signal
from core.pipeline import SAMPLE_RATE, FRAME_LENGTH, FRAME_COUNT, DEFAULT_NUM_MEL_FILTERS


def frame_signal(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    samples = np.asarray(signal, dtype=np.float32).reshape(-1)
    if samples.size != FRAME_COUNT * FRAME_LENGTH:
        raise ValueError(f"signal must contain exactly {FRAME_COUNT * FRAME_LENGTH} samples")

    frames = samples.reshape(FRAME_COUNT, FRAME_LENGTH)
    return np.asarray(frames, dtype=np.float32)


def _hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def _mel_to_hz(mel_values: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def _dct_basis(num_coefficients: int, num_filters: int) -> np.ndarray:
    filter_indices = np.arange(num_filters, dtype=np.float32)
    coefficient_indices = np.arange(num_coefficients, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / num_filters * (filter_indices + 0.5) * coefficient_indices)
    basis[0] *= np.sqrt(1.0 / num_filters)
    if num_coefficients > 1:
        basis[1:] *= np.sqrt(2.0 / num_filters)
    return basis.astype(np.float32, copy=False)


def _mel_filterbank(sample_rate: int, n_fft: int, num_mel_filters: int) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(np.asarray(0.0, dtype=np.float32)),
        _hz_to_mel(np.asarray(sample_rate / 2.0, dtype=np.float32)),
        num_mel_filters + 2,
        dtype=np.float32,
    )
    hz_points = _mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    filterbank = np.zeros((num_mel_filters, n_fft // 2 + 1), dtype=np.float32)
    for filter_index in range(num_mel_filters):
        left = bin_indices[filter_index]
        center = bin_indices[filter_index + 1]
        right = bin_indices[filter_index + 2]

        if center > left:
            filterbank[filter_index, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / max(center - left, 1)
        if right > center:
            filterbank[filter_index, center:right] = (
                right - np.arange(center, right, dtype=np.float32)
            ) / max(right - center, 1)
    return filterbank


def compute_mfcc(
    frames: Sequence[Sequence[float]] | np.ndarray,
    num_coefficients: int = 10,
    sample_rate: int = SAMPLE_RATE,
    num_mel_filters: int = DEFAULT_NUM_MEL_FILTERS,
    use_hamming_window: bool = True,
) -> np.ndarray:
    framed_signal = np.asarray(frames, dtype=np.float32)
    if framed_signal.shape != (FRAME_COUNT, FRAME_LENGTH):
        raise ValueError(f"frames must have shape ({FRAME_COUNT}, {FRAME_LENGTH})")
    if num_coefficients <= 0:
        raise ValueError("num_coefficients must be positive")

    frame_length = framed_signal.shape[1]
    n_fft = frame_length
    if use_hamming_window:
        window = np.hamming(frame_length).astype(np.float32)
        framed_signal = framed_signal * window

    spectrum = np.fft.rfft(framed_signal, n=n_fft, axis=1)
    power_spectrum = (np.abs(spectrum) ** 2) / float(n_fft)

    mel_filterbank = _mel_filterbank(sample_rate, n_fft, num_mel_filters)
    mel_energies = power_spectrum @ mel_filterbank.T
    log_mel_energies = np.log(np.maximum(mel_energies, np.finfo(np.float32).eps))

    basis = _dct_basis(num_coefficients, num_mel_filters)
    mfcc_matrix = log_mel_energies @ basis.T
    return mfcc_matrix.astype(np.float32, copy=False).reshape(-1)


def extract_features(raw_samples: Sequence[int] | np.ndarray, n_mfcc: int = 10) -> np.ndarray:
    signal = preprocess_signal(raw_samples)
    frames = frame_signal(signal)
    return compute_mfcc(frames, num_coefficients=n_mfcc)


def extract_feature_matrix(samples: Sequence[Sequence[int] | np.ndarray], n_mfcc: int = 10) -> np.ndarray:
    feature_rows = [extract_features(sample, n_mfcc=n_mfcc) for sample in samples]
    return np.vstack(feature_rows)
