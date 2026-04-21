from __future__ import annotations

from pathlib import Path

import numpy as np

from common.paths import feature_root, time_series_root
from project_shared.frequency import normalize_frequency
from project_shared.targets import compute_forecast_window


DEFAULT_VALID_SIZE = {"daily": 60, "monthly": 6}
DEFAULT_TEST_SIZE = {"daily": 60, "monthly": 6}


def _index_filename(prefix: str, window_dir: Path) -> Path:
    preferred = window_dir / f"{prefix}_index.npy"
    legacy = window_dir / f"{prefix}_months.npy"
    return preferred if preferred.exists() else legacy


def load_split_arrays(window_length: int, root_path: Path | None = None, frequency: str = "monthly") -> dict:
    frequency = normalize_frequency(frequency)
    root = Path(root_path) if root_path is not None else time_series_root("fusion", frequency)
    window_dir = root / f"window_{window_length}"
    bundles = {}
    for split_name, prefix in (("train", "train"), ("valid", "valid"), ("test", "test")):
        index_path = _index_filename(prefix, window_dir)
        index_values = np.load(index_path, allow_pickle=True)
        bundles[split_name] = {
            "features": np.load(window_dir / f"{prefix}.npy").astype(np.float32),
            "labels": np.load(window_dir / f"{prefix}_labels.npy").astype(np.float32),
            "months": index_values,
            "index": index_values,
        }
    return bundles


def combine_splits(split_arrays: dict) -> dict:
    index_values = np.concatenate(
        [split_arrays["train"]["index"], split_arrays["valid"]["index"], split_arrays["test"]["index"]],
        axis=0,
    )
    return {
        "features": np.concatenate(
            [split_arrays["train"]["features"], split_arrays["valid"]["features"], split_arrays["test"]["features"]],
            axis=0,
        ),
        "labels": np.concatenate(
            [split_arrays["train"]["labels"], split_arrays["valid"]["labels"], split_arrays["test"]["labels"]],
            axis=0,
        ),
        "months": index_values,
        "index": index_values,
    }


def infer_input_dim(window_length: int, root_path: Path | None = None, frequency: str = "monthly") -> int:
    frequency = normalize_frequency(frequency)
    root = Path(root_path) if root_path is not None else time_series_root("fusion", frequency)
    train_array = np.load(root / f"window_{window_length}" / "train.npy")
    if train_array.ndim != 3:
        raise ValueError(f"Expected [N, T, D] input array, got {train_array.shape}.")
    return int(train_array.shape[-1])


def build_reference_lookup(frequency: str = "monthly") -> dict[str, float]:
    frequency = normalize_frequency(frequency)
    structured_root = feature_root("structured", frequency)
    index_path = structured_root / "structured_index.npy"
    if not index_path.exists():
        index_path = structured_root / "structured_months.npy"
    index_values = np.load(index_path, allow_pickle=True)
    reference = np.load(structured_root / "structured_reference.npy").astype(np.float32)
    if len(index_values) != len(reference):
        raise ValueError("Structured index and reference arrays must have the same length.")
    return {str(index_values[idx]): float(reference[idx]) for idx in range(len(index_values))}


def attach_reference(index_values: np.ndarray, lookup: dict[str, float]) -> np.ndarray:
    return np.asarray([lookup[str(value)] for value in index_values], dtype=np.float32)


def with_reference(split_arrays: dict, lookup: dict[str, float] | None = None, frequency: str = "monthly") -> dict:
    reference_lookup = lookup or build_reference_lookup(frequency)
    bundles = {}
    for split_name, bundle in split_arrays.items():
        bundles[split_name] = {
            **bundle,
            "reference": attach_reference(bundle["index"], reference_lookup),
        }
    return bundles


def slice_bundle(bundle: dict, start: int, end: int) -> dict:
    index_values = bundle["index"][start:end]
    return {
        "features": bundle["features"][start:end],
        "labels": bundle["labels"][start:end],
        "months": index_values,
        "index": index_values,
        "reference": bundle["reference"][start:end],
    }


def build_rolling_plan(
    combined_bundle: dict,
    val_size: int | None = None,
    test_size: int | None = None,
    rolling_folds: int = 3,
    frequency: str = "monthly",
) -> dict:
    frequency = normalize_frequency(frequency)
    val_size = int(val_size or DEFAULT_VALID_SIZE[frequency])
    test_size = int(test_size or DEFAULT_TEST_SIZE[frequency])
    total_samples = len(combined_bundle["labels"])
    pre_test_end = total_samples - test_size
    first_fold_train_end = pre_test_end - rolling_folds * val_size
    if first_fold_train_end <= 0:
        raise ValueError(
            f"Not enough samples ({total_samples}) for rolling validation: "
            f"val_size={val_size}, test_size={test_size}, folds={rolling_folds}."
        )

    validation_folds = []
    for fold_idx in range(rolling_folds):
        train_end = first_fold_train_end + fold_idx * val_size
        valid_start = train_end
        valid_end = valid_start + val_size
        validation_folds.append(
            {
                "fold_index": fold_idx + 1,
                "train": slice_bundle(combined_bundle, 0, train_end),
                "valid": slice_bundle(combined_bundle, valid_start, valid_end),
            }
        )

    final_valid_start = pre_test_end - val_size
    final_valid_end = pre_test_end
    final_plan = {
        "train": slice_bundle(combined_bundle, 0, final_valid_start),
        "valid": slice_bundle(combined_bundle, final_valid_start, final_valid_end),
        "test": slice_bundle(combined_bundle, pre_test_end, total_samples),
    }
    return {
        "validation_folds": validation_folds,
        "final_plan": final_plan,
        "val_size": val_size,
        "test_size": test_size,
        "rolling_folds": rolling_folds,
    }


def compute_future_window(last_index: str, frequency: str = "monthly", horizon_days: int = 7) -> dict[str, str]:
    return compute_forecast_window(last_index, normalize_frequency(frequency), horizon_days=horizon_days)


def compute_future_month(last_month: str) -> str:
    forecast_window = compute_future_window(last_month, frequency="monthly")
    return forecast_window["forecast_start_date"][:7]


def _load_variant_feature_bank(input_variant: str, frequency: str = "monthly") -> tuple[np.ndarray, np.ndarray]:
    frequency = normalize_frequency(frequency)
    variant_files = {
        "fusion": ("fusion", "fusion_features.npy", "fusion_index.npy", "fusion_months.npy"),
        "text": ("text", "text_embeddings.npy", "text_index.npy", "text_months.npy"),
        "image": ("image", "image_embeddings.npy", "image_index.npy", "image_months.npy"),
        "structured": ("structured", "structured_features.npy", "structured_index.npy", "structured_months.npy"),
    }
    if input_variant not in variant_files:
        raise ValueError(f"Unsupported input_variant: {input_variant}")
    kind, feature_name, index_name, legacy_index_name = variant_files[input_variant]
    root = feature_root(kind, frequency)
    features = np.load(root / feature_name).astype(np.float32)
    index_path = root / index_name
    if not index_path.exists():
        index_path = root / legacy_index_name
    index_values = np.load(index_path, allow_pickle=True)
    return features, index_values


def load_latest_variant_window(
    window_length: int,
    input_variant: str = "fusion",
    frequency: str = "monthly",
) -> tuple[np.ndarray, np.ndarray]:
    features, index_values = _load_variant_feature_bank(input_variant, frequency)
    if len(features) < window_length:
        raise ValueError(f"Not enough features for input_variant={input_variant} window_length={window_length}.")
    return features[-window_length:], index_values


def load_latest_fusion_window(window_length: int, frequency: str = "monthly") -> tuple[np.ndarray, np.ndarray]:
    return load_latest_variant_window(window_length=window_length, input_variant="fusion", frequency=frequency)
