from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .dataset import load_dataset
from .features import extract_feature_matrix


@dataclass(slots=True)
class ModelBundle:
    model: DecisionTreeClassifier
    class_names: list[str]
    n_mfcc: int
    max_depth: int
    min_samples_split: int
    accuracy: float
    report: str
    confusion: np.ndarray


def _dataset_to_matrix(dataset: dict[str, list[list[float]]], n_mfcc: int, expected_class_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if expected_class_names is not None:
        class_names = expected_class_names
    else:
        class_names = [label for label, samples in sorted(dataset.items()) if samples]

    raw_samples: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, label in enumerate(class_names):
        samples = dataset.get(label, [])
        for sample in samples:
            raw_samples.append(np.asarray(sample, dtype=np.float32))
            labels.append(class_index)

    if not raw_samples:
        raise ValueError("dataset is empty")

    features = extract_feature_matrix(raw_samples, n_mfcc=n_mfcc)
    return features, np.asarray(labels, dtype=np.int64), class_names


def train_from_dataset(
    dataset: dict[str, list[list[float]]],
    n_mfcc: int = 10,
    max_depth: int = 8,
    min_samples_split: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelBundle:
    features, labels, class_names = _dataset_to_matrix(dataset, n_mfcc=n_mfcc)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, target_names=class_names, zero_division=0)
    confusion = confusion_matrix(y_test, predictions)
    return ModelBundle(
        model=model,
        class_names=class_names,
        n_mfcc=n_mfcc,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        accuracy=accuracy,
        report=report,
        confusion=confusion,
    )


def autotune_from_datasets(
    training_dataset: dict[str, list[list[float]]],
    validation_dataset: dict[str, list[list[float]]] | None = None,
    n_mfcc_values: Sequence[int] = (6, 8, 10),
    max_depth_values: Sequence[int] = (3, 5, 7, 10),
    min_samples_split_values: Sequence[int] = (2, 5, 10),
    random_state: int = 42,
) -> tuple[ModelBundle, list[tuple[int, int, int, float]]]:
    rows: list[tuple[int, int, int, float]] = []
    best_bundle: ModelBundle | None = None
    best_score = -1.0
    train_dataset = training_dataset
    val_dataset = validation_dataset or training_dataset

    for n_mfcc in n_mfcc_values:
        train_features, train_labels, class_names = _dataset_to_matrix(train_dataset, n_mfcc=n_mfcc)
        val_features, val_labels, _ = _dataset_to_matrix(val_dataset, n_mfcc=n_mfcc, expected_class_names=class_names)
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                model = DecisionTreeClassifier(
                    random_state=random_state,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                model.fit(train_features, train_labels)
                predictions = model.predict(val_features)
                score = float(accuracy_score(val_labels, predictions))
                rows.append((n_mfcc, max_depth, min_samples_split, score))
                if score > best_score:
                    best_score = score
                    best_bundle = ModelBundle(
                        model=model,
                        class_names=class_names,
                        n_mfcc=n_mfcc,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        accuracy=score,
                        report=classification_report(val_labels, predictions, target_names=class_names, zero_division=0),
                        confusion=confusion_matrix(val_labels, predictions),
                    )

    if best_bundle is None:
        raise RuntimeError("no configurations were evaluated")
    return best_bundle, rows


def evaluate_offline(
    model: DecisionTreeClassifier,
    dataset: dict[str, list[list[float]]],
    n_mfcc: int,
    expected_class_names: list[str] | None = None,
) -> tuple[float, str, np.ndarray]:
    features, labels, class_names = _dataset_to_matrix(dataset, n_mfcc=n_mfcc, expected_class_names=expected_class_names)
    predictions = model.predict(features)
    accuracy = float(accuracy_score(labels, predictions))
    report = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    confusion = confusion_matrix(labels, predictions)
    return accuracy, report, confusion


def save_model_artifact(path: str | Path, bundle: ModelBundle) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "class_names": bundle.class_names,
            "n_mfcc": bundle.n_mfcc,
            "max_depth": bundle.max_depth,
            "min_samples_split": bundle.min_samples_split,
        },
        target,
    )


def load_model_artifact(path: str | Path):
    return joblib.load(path)


def predict_command(model, feature_vector: Sequence[float] | np.ndarray, class_names: Sequence[str]) -> tuple[int, str]:
    vector = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    predicted_index = int(model.predict(vector)[0])

    # --- DISTANCE REJECTION LOGIC (COMMENTED OUT FOR NOW) ---
    # if centroids is not None and thresholds is not None:
    #     centroid = centroids[predicted_index]
    #     distance = float(np.linalg.norm(vector[0] - centroid))
    #     print(f"  [DEBUG] distance to '{class_names[predicted_index]}': {distance:.2f} (threshold: {thresholds[predicted_index]:.2f})")
    #     if distance > thresholds[predicted_index]:
    #         return -1, "unknown"

    if 0 <= predicted_index < len(class_names):
        return predicted_index, str(class_names[predicted_index])
    return predicted_index, str(predicted_index)


def run_autotune(training_path: str | Path, output_model_path: str | Path, test_path: str | None = None) -> ModelBundle:
    training = load_dataset(training_path)
    validation = load_dataset(test_path) if test_path is not None else None
    best, rows = autotune_from_datasets(training, validation)
    save_model_artifact(output_model_path, best)
    return best

# --- DISTANCE REJECTION LOGIC (COMMENTED OUT FOR NOW) ---
# def _compute_rejection_params(features: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
#     centroids = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
#     thresholds = np.zeros(num_classes, dtype=np.float32)
#     for c in range(num_classes):
#         class_features = features[labels == c]
#         if class_features.shape[0] > 0:
#             centroid = class_features.mean(axis=0)
#             centroids[c] = centroid
#             distances = np.linalg.norm(class_features - centroid, axis=1)
#             thresholds[c] = float(np.percentile(distances, 95) * 2.0)
#     return centroids, thresholds
