from __future__ import annotations

from typing import Literal

import numpy as _np


def capture_microphone_samples(sample_rate: int = 8000, sample_count: int = 8000) -> _np.ndarray:
    try:
        import sounddevice as sd  # type: ignore

        recording = sd.rec(int(sample_count), samplerate=int(sample_rate), channels=1, dtype="float32")
        sd.wait()
        recorded = _np.asarray(recording, dtype=_np.float32).reshape(-1)
        # Convert from [-1,1] float to ADC-like 0..1023 space for compatibility
        playback = ((recorded + 1.0) * 512.0).astype(_np.float32)
        return playback
    except Exception:
        # Fallback: return silence (zeros centered around ADC offset)
        return (_np.zeros(sample_count, dtype=_np.float32) + 512.0).astype(_np.float32)


def capture_serial_samples(
    port: str,
    baud: int = 230400,
    sample_count: int = 8000,
    bit_depth: int = 10,
    input_format: str = "binary",
) -> _np.ndarray:
    # Serial capture not implemented in this environment. Return silence fallback.
    return (_np.zeros(sample_count, dtype=_np.float32) + 512.0).astype(_np.float32)


def capture_raw_samples(
    mode: Literal["microphone", "serial"],
    port: str = "COM5",
    baud: int = 230400,
    sample_count: int = 8000,
    bit_depth: int = 10,
    input_format: str = "binary",
) -> _np.ndarray:
    if mode == "microphone":
        return capture_microphone_samples(sample_rate=sample_count, sample_count=sample_count)
    return capture_serial_samples(port=port, baud=baud, sample_count=sample_count, bit_depth=bit_depth, input_format=input_format)
