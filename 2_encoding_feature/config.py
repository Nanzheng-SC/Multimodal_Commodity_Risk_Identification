#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from project_shared.frequency import (
    SUPPORTED_FREQUENCIES,
    default_window_length,
    default_window_lengths,
    index_column_for_frequency,
    normalize_frequency,
)
from project_shared.paths import (
    ENCODING_OUTPUT_ROOT,
    EVENT_MANIFEST_PATH,
    IMAGE_THUMBNAIL_DIR,
    feature_root,
    preferred_image_manifest_path,
    preferred_text_input_path,
    structured_source_path,
    time_series_root,
)

# Raw data paths
DATA_DIR = os.path.join(BASE_DIR, "1_data_handling", "raw")
TEXT_DATA_PATH = str(preferred_text_input_path())
IMAGE_MANIFEST_PATH = str(preferred_image_manifest_path())
IMAGE_DIR = str(IMAGE_THUMBNAIL_DIR)
EVENT_DATA_PATH = str(EVENT_MANIFEST_PATH)
TEXT_BERT_MODELS_DIR = os.path.join(BASE_DIR, "2_encoding_feature", "text_bert", "models")
MULTILINGUAL_BERT_DIR = os.path.join(TEXT_BERT_MODELS_DIR, "bert-base-multilingual-cased")


def structured_data_path(frequency: str = "monthly") -> str:
    frequency = normalize_frequency(frequency)
    return str(structured_source_path(frequency))


def output_root(frequency: str = "monthly") -> str:
    return str(feature_root("..", frequency).parent.resolve())


def features_dir(kind: str, frequency: str = "monthly") -> str:
    return str(feature_root(kind, frequency))


def time_series_dir(variant: str, frequency: str = "monthly") -> str:
    return str(time_series_root(variant, frequency))


def index_column_name(frequency: str = "monthly") -> str:
    return index_column_for_frequency(frequency)


# Backward-compatible monthly defaults
STRUCTURED_DATA_PATH = structured_data_path("monthly")
OUTPUT_DIR = str(ENCODING_OUTPUT_ROOT)
STRUCTURED_FEATURES_DIR = features_dir("structured", "monthly")
TEXT_FEATURES_DIR = features_dir("text", "monthly")
IMAGE_FEATURES_DIR = features_dir("image", "monthly")
FUSION_FEATURES_DIR = features_dir("fusion", "monthly")
TIME_SERIES_DIR = time_series_dir("fusion", "monthly")
TIME_SERIES_TEXT_DIR = time_series_dir("text", "monthly")
TIME_SERIES_IMAGE_DIR = time_series_dir("image", "monthly")
TIME_SERIES_STRUCTURED_DIR = time_series_dir("structured", "monthly")

# Structured columns removed before downstream modeling.
STRUCTURED_DROP_COLUMNS = []

# Model configuration
BERT_MODEL_NAME = (
    MULTILINGUAL_BERT_DIR
    if os.path.isdir(MULTILINGUAL_BERT_DIR)
    else "bert-base-multilingual-cased"
)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Text encoding
MAX_TEXT_LENGTH = 512
TEXT_SUMMARY_LENGTH = 1000

# Image encoding
IMAGE_SIZE = (224, 224)
LIGHT_IMAGE_SIZE = (768, 768)
LIGHT_IMAGE_WEBP_QUALITY = 82

# Fusion
UNIFIED_DIM = 256

# Time-series windows
WINDOW_LENGTHS_BY_FREQUENCY = {
    "daily": default_window_lengths("daily"),
    "monthly": default_window_lengths("monthly"),
}
DEFAULT_WINDOW_LENGTH_BY_FREQUENCY = {
    "daily": default_window_length("daily"),
    "monthly": default_window_length("monthly"),
}
WINDOW_LENGTHS = WINDOW_LENGTHS_BY_FREQUENCY["monthly"]
DEFAULT_WINDOW_LENGTH = DEFAULT_WINDOW_LENGTH_BY_FREQUENCY["monthly"]
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Runtime
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def validate_frequency(frequency: str) -> str:
    return normalize_frequency(frequency)


def is_supported_frequency(frequency: str) -> bool:
    return str(frequency).strip().lower() in SUPPORTED_FREQUENCIES
