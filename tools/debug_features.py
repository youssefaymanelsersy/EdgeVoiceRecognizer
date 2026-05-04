from __future__ import annotations

from pathlib import Path

from core.preprocess import preprocess_signal
from core.features import frame_signal, compute_mfcc
import numpy as np
import matplotlib.pyplot as plt


def debug_feature_file(path: str | Path) -> None:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        raw = np.asarray(np.load(p), dtype=np.float32).reshape(-1)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
        tokens = text.replace(",", " ").replace(";", " ").split()
        raw = np.asarray([float(t) for t in tokens], dtype=np.float32)

    signal = preprocess_signal(raw)
    frames = frame_signal(signal)
    mfcc_vector = compute_mfcc(frames)
    coefficient_count = mfcc_vector.size // frames.shape[0]
    mfcc_matrix = mfcc_vector.reshape(frames.shape[0], coefficient_count)

    print(f"Feature matrix shape: {mfcc_matrix.shape}")
    # Simple plot
    plt.imshow(mfcc_matrix.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.title("MFCC Heatmap")
    plt.xlabel("Frame index")
    plt.ylabel("Coefficient index")
    plt.show()

