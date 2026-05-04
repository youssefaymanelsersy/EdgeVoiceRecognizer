from __future__ import annotations

from pathlib import Path

from core.audio import capture_raw_samples
from core.dataset import (
    DATA_DIR,
    MODELS_DIR,
    SAMPLES_FILE,
    TEST_SAMPLES_FILE,
    DEFAULT_COMMANDS,
    add_sample,
    create_empty_dataset,
    dataset_summary,
    load_dataset,
    replace_sample,
    save_dataset,
)
from core.features import extract_features
from core.model import (
    ModelBundle,
    evaluate_offline,
    load_model_artifact,
    save_model_artifact,
    train_from_dataset,
    run_autotune,
)
from core.export import export_tree_to_c
from tools import debug_feature_file, inspect_dataset, live_replay
from config import CAPTURE_MODE, SERIAL_PORT, SERIAL_BAUDRATE


def _input_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def _input_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else default


def _input_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    default = options[default_index]
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _capture_mode() -> str:
    return CAPTURE_MODE


def _capture_kwargs() -> dict[str, object]:
    return {
        "port": SERIAL_PORT,
        "baud": SERIAL_BAUDRATE,
        "bit_depth": 10,
        "input_format": "binary",
    }


def _load_or_create_dataset(path: Path):
    if path.exists():
        return load_dataset(path)
    return create_empty_dataset(DEFAULT_COMMANDS)


def _record_many(dataset_path: Path, prompt_title: str, overwrite: bool = False) -> None:
    # recordings_per_command will be provided by caller via prompt_title convention
    # prompt_title indicates whether training (20) or test (10)
    recordings_per_command = 20 if "train" in prompt_title.lower() or "dataset" in prompt_title.lower() and "test" not in prompt_title.lower() else 10
    dataset = create_empty_dataset(DEFAULT_COMMANDS) if overwrite else _load_or_create_dataset(dataset_path)
    mode = _capture_mode()
    capture_kwargs = _capture_kwargs()
    print(f"Recording mode: {mode}")
    for label in DEFAULT_COMMANDS:
        print(f"\n{prompt_title}: {label}")
        dataset.setdefault(label, [])
        if overwrite:
            dataset[label] = []
        while len(dataset[label]) < recordings_per_command:
            arm = input(f"Press Enter to start recording {len(dataset[label]) + 1}/{recordings_per_command} for {label}, or q to quit: ").strip().lower()
            if arm.startswith("q"):
                save_dataset(dataset_path, dataset)
                return

            print(f"Recording {len(dataset[label]) + 1}/{recordings_per_command} for {label}")
            raw_samples = capture_raw_samples(mode=mode, **capture_kwargs)
            if raw_samples.size != 8000:
                print(f"Rejected recording: expected 8000 samples, got {raw_samples.size}")
                continue
            print(f"min={raw_samples.min():.2f}, max={raw_samples.max():.2f}, mean={raw_samples.mean():.2f}")
            choice = input("Save this sample? [Y/n/r/q]: ").strip().lower()
            if choice.startswith("q"):
                save_dataset(dataset_path, dataset)
                return
            if choice.startswith("r"):
                continue
            add_sample(dataset, label, raw_samples)
            print("saved")
    save_dataset(dataset_path, dataset)
    print(f"Saved dataset to {dataset_path}")


def _train_and_save(dataset_path: Path, model_path: Path) -> ModelBundle:
    dataset = load_dataset(dataset_path)
    bundle = train_from_dataset(dataset, n_mfcc=10, max_depth=8, min_samples_split=2, test_size=0.25)
    save_model_artifact(model_path, bundle)
    print(f"accuracy: {bundle.accuracy:.4f}")
    print(bundle.report)
    print(bundle.confusion)
    return bundle


def _rebuild_model() -> None:
    if not SAMPLES_FILE.exists():
        print(f"Error: training dataset missing: {SAMPLES_FILE}")
        return
    summary = dataset_summary(load_dataset(SAMPLES_FILE))
    if sum(summary.values()) == 0:
        print(f"Error: training dataset is empty: {SAMPLES_FILE}")
        return
    _train_and_save(SAMPLES_FILE, MODELS_DIR / "model.joblib")


def _run_live() -> None:
    model_file = Path(MODELS_DIR / "model.joblib")
    if not model_file.exists():
        print("Model missing. Please train the model first (Rebuild model).")
        return
    artifact = load_model_artifact(model_file)
    model = artifact["model"]
    class_names = artifact.get("class_names", DEFAULT_COMMANDS)
    n_mfcc = int(artifact.get("n_mfcc", 10))
    mode = _capture_mode()
    print("Live recognition. Press Enter to capture a sample, Ctrl+C to exit.")
    try:
        while True:
            input()
            raw_samples = capture_raw_samples(mode=mode, **_capture_kwargs())
            if raw_samples.size != 8000:
                print(f"Rejected recording: expected 8000 samples, got {raw_samples.size}")
                continue
            features = extract_features(raw_samples, n_mfcc=n_mfcc)
            from core.model import predict_command
            idx, name = predict_command(model, features, class_names)
            if idx == -1:
                print(f"unknown (distance rejection)")
            else:
                print(name)
    except KeyboardInterrupt:
        print("\nLive recognition stopped by user.")


def _evaluate_offline() -> None:
    model_file = Path(MODELS_DIR / "model.joblib")
    if not model_file.exists():
        print("Model missing. Please train the model first (Rebuild model).")
        return
    if not TEST_SAMPLES_FILE.exists():
        print(f"Error: test dataset missing: {TEST_SAMPLES_FILE}")
        return
    artifact = load_model_artifact(model_file)
    accuracy, report, confusion = evaluate_offline(
        artifact["model"],
        load_dataset(TEST_SAMPLES_FILE),
        int(artifact.get("n_mfcc", 10)),
        expected_class_names=artifact.get("class_names", DEFAULT_COMMANDS)
    )
    print(f"accuracy: {accuracy:.4f}")
    print(report)
    print(confusion)
    # Per-class accuracy
    import numpy as _np

    row_sums = confusion.sum(axis=1).astype(float)
    per_class = []
    for idx, row_sum in enumerate(row_sums):
        if row_sum == 0:
            per_class.append(None)
        else:
            per_class.append(confusion[idx, idx] / row_sum)
    for idx, acc in enumerate(per_class):
        name = artifact.get("class_names", DEFAULT_COMMANDS)[idx] if idx < len(artifact.get("class_names", DEFAULT_COMMANDS)) else str(idx)
        if acc is None:
            print(f"  {name}: no samples")
        else:
            print(f"  {name}: {acc:.4f}")


def _autotune() -> None:
    if not SAMPLES_FILE.exists():
        print(f"Error: training dataset missing: {SAMPLES_FILE}")
        return
    if not TEST_SAMPLES_FILE.exists():
        print(f"Error: test dataset missing: {TEST_SAMPLES_FILE}. Autotune requires separate test dataset.")
        return
    # Use run_autotune helper but provide explicit test dataset path to ensure testing on separate data
    print("\nRunning autotune across configurations... Please wait.")
    bundle = run_autotune(str(SAMPLES_FILE), str(MODELS_DIR / "model.joblib"), str(TEST_SAMPLES_FILE))
    
    print("\n--- Autotune Complete ---")
    print(f"Best Configuration:")
    print(f"  MFCC Coefficients  : {bundle.n_mfcc}")
    print(f"  Max Depth          : {bundle.max_depth}")
    print(f"  Min Samples Split  : {bundle.min_samples_split}")
    print(f"Validation Accuracy: {bundle.accuracy:.4f}")
    print("\nClassification Report:")
    print(bundle.report)


def _export_model() -> None:
    model_file = Path(MODELS_DIR / "model.joblib")
    if not model_file.exists():
        print("Model missing. Please train the model first (Rebuild model).")
        return
    artifact = load_model_artifact(model_file)
    code = export_tree_to_c(artifact["model"], artifact.get("class_names", DEFAULT_COMMANDS))
    include_dir = Path("include")
    include_dir.mkdir(parents=True, exist_ok=True)
    output_file = include_dir / "model.h"
    output_file.write_text(code, encoding="utf-8")
    print(f"Saved C export to {output_file}")


def _retake_samples(target_file: Path, label_prompt: str) -> None:
    # Deprecated: use dedicated retake functions in strict workflow
    raise RuntimeError("Use dedicated retake functions: _retake_training or _retake_test")


def _retake_training() -> None:
    if not SAMPLES_FILE.exists():
        print(f"Error: training dataset missing: {SAMPLES_FILE}")
        return
    dataset = load_dataset(SAMPLES_FILE)
    label = input("Which command to retake (name): ").strip() or DEFAULT_COMMANDS[0]
    if label not in dataset:
        print(f"Unknown label: {label}")
        return
    # Delete its 20 samples
    dataset[label] = []
    mode = _capture_mode()
    capture_kwargs = _capture_kwargs()
    recordings_per_command = 20
    print(f"Retaking training samples for: {label}")
    while len(dataset[label]) < recordings_per_command:
        arm = input(f"Press Enter to start recording {len(dataset[label]) + 1}/{recordings_per_command} for {label}, or q to quit: ").strip().lower()
        if arm.startswith("q"):
            save_dataset(SAMPLES_FILE, dataset)
            return
        raw_samples = capture_raw_samples(mode=mode, **capture_kwargs)
        if raw_samples.size != 8000:
            print(f"Rejected recording: expected 8000 samples, got {raw_samples.size}")
            continue
        add_sample(dataset, label, raw_samples)
        print("saved")
    save_dataset(SAMPLES_FILE, dataset)
    print(f"Updated training dataset: {SAMPLES_FILE} -> {label}")


def _retake_test() -> None:
    if not TEST_SAMPLES_FILE.exists():
        print(f"Error: test dataset missing: {TEST_SAMPLES_FILE}")
        return
    dataset = load_dataset(TEST_SAMPLES_FILE)
    label = input("Which command to retake (name): ").strip() or DEFAULT_COMMANDS[0]
    if label not in dataset:
        print(f"Unknown label: {label}")
        return
    # Delete its 10 samples
    dataset[label] = []
    mode = _capture_mode()
    capture_kwargs = _capture_kwargs()
    recordings_per_command = 10
    print(f"Retaking test samples for: {label}")
    while len(dataset[label]) < recordings_per_command:
        arm = input(f"Press Enter to start recording {len(dataset[label]) + 1}/{recordings_per_command} for {label}, or q to quit: ").strip().lower()
        if arm.startswith("q"):
            save_dataset(TEST_SAMPLES_FILE, dataset)
            return
        raw_samples = capture_raw_samples(mode=mode, **capture_kwargs)
        if raw_samples.size != 8000:
            print(f"Rejected recording: expected 8000 samples, got {raw_samples.size}")
            continue
        add_sample(dataset, label, raw_samples)
        print("saved")
    save_dataset(TEST_SAMPLES_FILE, dataset)
    print(f"Updated test dataset: {TEST_SAMPLES_FILE} -> {label}")


def _tools_menu() -> None:
    while True:
        print("\nTools")
        print("1. Replay recorded audio")
        print("2. Debug features")
        print("3. Inspect dataset")
        print("0. Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return
        if choice == "1":
            live_replay(mode=_capture_mode(), **_capture_kwargs())
        elif choice == "2":
            debug_feature_file(input("Path to sample file: ").strip())
        elif choice == "3":
            inspect_dataset(input("Path to dataset file: ").strip() or str(SAMPLES_FILE))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print("\nVoice Command Recognition System")
        print("1. Record dataset (once)")
        print("2. Rebuild model (no recording)")
        print("3. Run (live recognition)")
        print("4. Autotune model")
        print("5. Evaluate model offline")
        print("6. Retake training samples")
        print("7. Retake test samples")
        print("8. Export model to C")
        print("9. Tools (Replay / Debug)")
        print("10. Record test dataset (once)")
        print("11. Exit")

        choice = input("Select an option: ").strip()
        if choice == "11":
            return
        if choice == "1":
            ans = input("Append to existing dataset or overwrite? [A/o]: ").strip().lower()
            _record_many(SAMPLES_FILE, "Recording training dataset", overwrite=(ans == "o"))
        elif choice == "2":
            _rebuild_model()
        elif choice == "3":
            _run_live()
        elif choice == "4":
            _autotune()
        elif choice == "5":
            _evaluate_offline()
        elif choice == "6":
            _retake_training()
        elif choice == "7":
            _retake_test()
        elif choice == "8":
            _export_model()
        elif choice == "9":
            _tools_menu()
        elif choice == "10":
            ans = input("Append to existing dataset or overwrite? [A/o]: ").strip().lower()
            _record_many(TEST_SAMPLES_FILE, "Recording test dataset", overwrite=(ans == "o"))


if __name__ == "__main__":
    main()
