from __future__ import annotations

"""Shared preprocessing wrapper exposing the pipeline implementation.

This module provides the single preprocessing entrypoint under `core.` and
delegates to `core.pipeline.preprocess_signal`.
"""

from core.pipeline import preprocess_signal

__all__ = ["preprocess_signal"]
