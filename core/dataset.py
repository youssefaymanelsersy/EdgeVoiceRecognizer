from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np


DEFAULT_COMMANDS = ["on", "off", "up", "down", "left", "right", "start", "stop"]
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
INCLUDE_DIR = ROOT_DIR / "include"
SAMPLES_FILE = DATA_DIR / "samples.json"
TEST_SAMPLES_FILE = DATA_DIR / "test_samples.json"


def create_empty_dataset(command_names: Iterable[str] = DEFAULT_COMMANDS) -> dict[str, list[list[float]]]:
    return {name: [] for name in command_names}


def _normalize_samples(samples: np.ndarray | list[float] | list[int]) -> list[float]:
    return np.asarray(samples, dtype=np.float32).reshape(-1).astype(float).tolist()


def load_dataset(path: str | Path) -> dict[str, list[list[float]]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return create_empty_dataset()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "commands" in payload:
        commands = payload["commands"]
        return {str(label): [list(map(float, sample)) for sample in samples] for label, samples in commands.items()}
    if isinstance(payload, dict):
        return {str(label): [list(map(float, sample)) for sample in samples] for label, samples in payload.items()}
    raise ValueError(f"Unsupported dataset format in {dataset_path}")


def save_dataset(path: str | Path, dataset: dict[str, list[list[float]]]) -> None:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"commands": dataset}
    dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_sample(dataset: dict[str, list[list[float]]], label: str, samples: np.ndarray | list[float] | list[int]) -> None:
    dataset.setdefault(label, []).append(_normalize_samples(samples))


def replace_sample(dataset: dict[str, list[list[float]]], label: str, index: int, samples: np.ndarray | list[float] | list[int]) -> None:
    if label not in dataset:
        raise KeyError(label)
    dataset[label][index] = _normalize_samples(samples)


def iter_samples(dataset: dict[str, list[list[float]]]):
    for label in sorted(dataset):
        for index, sample in enumerate(dataset[label]):
            yield label, index, np.asarray(sample, dtype=np.float32)


def dataset_summary(dataset: dict[str, list[list[float]]]) -> dict[str, int]:
    return {label: len(samples) for label, samples in sorted(dataset.items())}
