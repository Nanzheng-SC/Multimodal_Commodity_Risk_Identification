from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEFAULT_WINDOW_LENGTH_BY_FREQUENCY,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    WINDOW_LENGTHS_BY_FREQUENCY,
    features_dir,
    time_series_dir,
    validate_frequency,
)


TIME_SERIES_VARIANTS = ("fusion", "text", "image", "structured")


def time_series_config(frequency: str = "monthly") -> dict:
    frequency = validate_frequency(frequency)
    return {
        "window_lengths": WINDOW_LENGTHS_BY_FREQUENCY[frequency],
        "default_window_length": DEFAULT_WINDOW_LENGTH_BY_FREQUENCY[frequency],
        "split_strategy": "fixed_horizon",
        "fixed_valid_size": 60 if frequency == "daily" else 6,
        "fixed_test_size": 60 if frequency == "daily" else 6,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
    }


def variant_input_paths(frequency: str = "monthly") -> dict:
    frequency = validate_frequency(frequency)
    return {
        "fusion": {
            "features": os.path.join(features_dir("fusion", frequency), "fusion_features.npy"),
            "labels": os.path.join(features_dir("fusion", frequency), "fusion_labels.npy"),
            "months": os.path.join(features_dir("fusion", frequency), "fusion_index.npy"),
        },
        "text": {
            "features": os.path.join(features_dir("text", frequency), "text_embeddings.npy"),
            "months": os.path.join(features_dir("text", frequency), "text_index.npy"),
            "labels": os.path.join(features_dir("structured", frequency), "structured_labels.npy"),
            "label_months": os.path.join(features_dir("structured", frequency), "structured_index.npy"),
        },
        "image": {
            "features": os.path.join(features_dir("image", frequency), "image_embeddings.npy"),
            "months": os.path.join(features_dir("image", frequency), "image_index.npy"),
            "labels": os.path.join(features_dir("structured", frequency), "structured_labels.npy"),
            "label_months": os.path.join(features_dir("structured", frequency), "structured_index.npy"),
        },
        "structured": {
            "features": os.path.join(features_dir("structured", frequency), "structured_features.npy"),
            "labels": os.path.join(features_dir("structured", frequency), "structured_labels.npy"),
            "months": os.path.join(features_dir("structured", frequency), "structured_index.npy"),
        },
    }


def variant_output_dirs(frequency: str = "monthly") -> dict:
    frequency = validate_frequency(frequency)
    return {
        "fusion": time_series_dir("fusion", frequency),
        "text": time_series_dir("text", frequency),
        "image": time_series_dir("image", frequency),
        "structured": time_series_dir("structured", frequency),
    }


# Backward-compatible monthly exports
TIME_SERIES_CONFIG = time_series_config("monthly")
VARIANT_INPUT_PATHS = variant_input_paths("monthly")
VARIANT_OUTPUT_DIRS = variant_output_dirs("monthly")
INPUT_PATHS = VARIANT_INPUT_PATHS["fusion"]
OUTPUT_CONFIG = {
    "features_dir": time_series_dir("fusion", "monthly"),
    "train_file": "train.npy",
    "valid_file": "valid.npy",
    "test_file": "test.npy",
    "train_labels_file": "train_labels.npy",
    "valid_labels_file": "valid_labels.npy",
    "test_labels_file": "test_labels.npy",
    "train_months_file": "train_index.npy",
    "valid_months_file": "valid_index.npy",
    "test_months_file": "test_index.npy",
}
