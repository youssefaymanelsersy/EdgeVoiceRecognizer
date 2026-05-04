from __future__ import annotations

import numpy as np

from .pipeline import SAMPLE_LENGTH


def check_signal_range(signal: np.ndarray) -> bool:
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size != SAMPLE_LENGTH:
        return False
    if not np.isfinite(arr).all():
        return False
    return True


def check_features_vector(vec: np.ndarray) -> bool:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return False
    if not np.isfinite(arr).all():
        return False
    return True
