from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import validate_frequency
from time_series.config import TIME_SERIES_VARIANTS, time_series_config, variant_input_paths, variant_output_dirs
from time_series.dataset_builder import DatasetBuilder
from time_series.window_builder import WindowBuilder


def _load_array(path, allow_pickle=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input file not found: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def load_variant_arrays(variant, frequency: str):
    config = variant_input_paths(frequency)[variant]
    features = _load_array(config["features"]).astype(np.float32)

    if variant in {"fusion", "structured"}:
        labels = _load_array(config["labels"]).astype(np.float32)
        months = _load_array(config["months"], allow_pickle=True)
        return features, labels, months

    feature_months = _load_array(config["months"], allow_pickle=True)
    labels = _load_array(config["labels"]).astype(np.float32)
    label_months = _load_array(config["label_months"], allow_pickle=True)

    if len(features) != len(labels):
        raise ValueError(f"Variant {variant} length mismatch: features={len(features)} labels={len(labels)}")
    if len(feature_months) != len(label_months):
        raise ValueError(
            f"Variant {variant} index length mismatch: feature_index={len(feature_months)} label_index={len(label_months)}"
        )
    if not np.array_equal(feature_months.astype(str), label_months.astype(str)):
        raise ValueError(f"Variant {variant} alignment mismatch between modality index and structured labels.")
    return features, labels, label_months


def run_time_series_pipeline(window_length=None, variant="fusion", frequency: str = "monthly"):
    frequency = validate_frequency(frequency)
    frequency_config = time_series_config(frequency)
    if window_length is None:
        window_length = frequency_config["default_window_length"]

    window_builder = WindowBuilder(
        window_lengths=frequency_config["window_lengths"],
        default_window_length=frequency_config["default_window_length"],
    )
    dataset_builder = DatasetBuilder()
    dataset_builder.split_strategy = frequency_config["split_strategy"]
    dataset_builder.fixed_valid_size = frequency_config["fixed_valid_size"]
    dataset_builder.fixed_test_size = frequency_config["fixed_test_size"]

    features, labels, months = load_variant_arrays(variant, frequency)
    X, y, window_months = window_builder.build_windows_with_months(features, labels, months, window_length)

    input_samples = len(features)
    valid_samples = len(X)
    skipped_samples = input_samples - valid_samples
    min_index = min(window_months) if len(window_months) else "N/A"
    max_index = max(window_months) if len(window_months) else "N/A"
    dataset = dataset_builder.split_dataset(X, y, window_months)

    output_dir = variant_output_dirs(frequency)[variant]
    window_dir = dataset_builder.save_dataset(dataset, output_dir, window_length)

    print("=== Time-series window summary ===")
    print(f"Frequency: {frequency}")
    print(f"Variant: {variant}")
    print(f"Input samples: {input_samples}")
    print(f"Valid window samples: {valid_samples}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Index coverage: {min_index} to {max_index}")
    print(f"Window length: {window_length}")
    print(f"Output shape: {X.shape}")
    print(f"Train size: {len(dataset['train']['X'])}")
    print(f"Valid size: {len(dataset['valid']['X'])}")
    print(f"Test size: {len(dataset['test']['X'])}")
    print(f"Saved to: {window_dir}")
    print("==================================")
    return dataset


def run_all_window_lengths(variants=None, frequency: str = "monthly"):
    frequency = validate_frequency(frequency)
    selected_variants = list(variants or TIME_SERIES_VARIANTS)
    frequency_config = time_series_config(frequency)
    active_window_dirs = {f"window_{window_length}" for window_length in frequency_config["window_lengths"]}
    for variant in selected_variants:
        output_dir = variant_output_dirs(frequency)[variant]
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                path = os.path.join(output_dir, name)
                if name.startswith("window_") and name not in active_window_dirs and os.path.isdir(path):
                    shutil.rmtree(path)
        for window_length in frequency_config["window_lengths"]:
            print(f"\n--- Building frequency={frequency}, variant={variant}, window={window_length} ---")
            run_time_series_pipeline(window_length=window_length, variant=variant, frequency=frequency)


def parse_args():
    parser = argparse.ArgumentParser(description="Build dual-frequency windowed time-series datasets.")
    parser.add_argument("--window-length", type=int, default=None, help="Optional single window length to build.")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(TIME_SERIES_VARIANTS),
        help="Comma-separated subset of fusion,text,image,structured.",
    )
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if args.window_length is None:
        run_all_window_lengths(variants=variants, frequency=args.frequency)
    else:
        for variant_name in variants:
            run_time_series_pipeline(window_length=args.window_length, variant=variant_name, frequency=args.frequency)
