from __future__ import annotations

import numpy as np


def live_replay(mode: str, **capture_kwargs) -> None:
    from core.audio import capture_raw_samples
    
    print("Press Enter to capture audio for replay, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        return

    raw_samples = capture_raw_samples(mode=mode, **capture_kwargs)
    print(f"Captured {raw_samples.size} raw samples.")
    print(f"min={raw_samples.min():.3f}, max={raw_samples.max():.3f}, mean={raw_samples.mean():.3f}")
    
    try:
        import sounddevice as sd  # type: ignore

        playback = np.clip(raw_samples.astype(np.float32), 0.0, 1023.0)
        playback = (playback - 512.0) / 512.0
        sd.play(playback, samplerate=8000)
        sd.wait()
    except Exception:
        print("Audio playback unavailable (sounddevice error).")

