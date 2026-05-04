# Embedded Voice Command System

`main.py` is the single user-facing entry point.

Run:

```bash
python main.py
```

The menu provides:

- Record dataset once
- Rebuild model without re-recording
- Live recognition
- Autotune model
- Offline evaluation
- Retake training samples
- Retake test samples
- Export model to C
- Tools for replay, feature debugging, and dataset inspection

The MFCC pipeline remains shared across training, recognition, tuning, and debugging.
