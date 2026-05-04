from __future__ import annotations

from pathlib import Path

from core.dataset import dataset_summary, load_dataset


def inspect_dataset(path: str | Path) -> None:
    dataset = load_dataset(path)
    summary = dataset_summary(dataset)
    total = sum(summary.values())
    print(f"Dataset: {path}")
    print(f"Total recordings: {total}")
    for label, count in summary.items():
        print(f"  {label}: {count}")
