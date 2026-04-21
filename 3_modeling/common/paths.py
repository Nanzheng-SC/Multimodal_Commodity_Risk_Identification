from __future__ import annotations

import sys
from pathlib import Path

MODELING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_shared.frequency import normalize_frequency
from project_shared.paths import (
    MODELING_COMPARISON_ROOT,
    MODELING_OFFICIAL_ROOT,
    MODELING_RESULTS_ROOT,
    MODELING_ROOT as SHARED_MODELING_ROOT,
    MODELING_SCRATCH_ROOT,
    PROJECT_ROOT as SHARED_PROJECT_ROOT,
    feature_root as shared_feature_root,
    official_model_dir as shared_official_model_dir,
    scratch_model_dir as shared_scratch_model_dir,
    time_series_root as shared_time_series_root,
    timemixer_model_name as shared_timemixer_model_name,
)

MODELING_ROOT = SHARED_MODELING_ROOT
PROJECT_ROOT = SHARED_PROJECT_ROOT

RESULTS_ROOT = MODELING_RESULTS_ROOT
OFFICIAL_RESULTS_ROOT = MODELING_OFFICIAL_ROOT
COMPARISON_RESULTS_ROOT = MODELING_COMPARISON_ROOT
SCRATCH_RESULTS_ROOT = MODELING_SCRATCH_ROOT

FUSION_FEATURES_ROOT = shared_feature_root("fusion", "monthly")
STRUCTURED_FEATURES_ROOT = shared_feature_root("structured", "monthly")
TEXT_FEATURES_ROOT = shared_feature_root("text", "monthly")
IMAGE_FEATURES_ROOT = shared_feature_root("image", "monthly")

FINAL_MODEL_COMPARISON_DIR = COMPARISON_RESULTS_ROOT / "monthly" / "final_models"
TIMEMIXER_MODALITY_COMPARISON_DIR = COMPARISON_RESULTS_ROOT / "monthly" / "timemixer_modalities"


def feature_root(kind: str, frequency: str = "monthly") -> Path:
    return shared_feature_root(kind, normalize_frequency(frequency))


def official_model_dir(model_name: str, window_length: int, frequency: str = "monthly") -> Path:
    return shared_official_model_dir(model_name, window_length, normalize_frequency(frequency))


def comparison_dir(window_length: int, frequency: str = "monthly") -> Path:
    return COMPARISON_RESULTS_ROOT / normalize_frequency(frequency) / f"window_{window_length}"


def final_model_comparison_dir(frequency: str = "monthly") -> Path:
    return COMPARISON_RESULTS_ROOT / normalize_frequency(frequency) / "final_models"


def scratch_model_dir(model_name: str, window_length: int, frequency: str = "monthly") -> Path:
    return shared_scratch_model_dir(model_name, window_length, normalize_frequency(frequency))


def time_series_root(input_variant: str = "fusion", frequency: str = "monthly") -> Path:
    return shared_time_series_root(input_variant, normalize_frequency(frequency))


def timemixer_model_name(input_variant: str = "fusion") -> str:
    return shared_timemixer_model_name(input_variant)


def timemixer_official_dir(window_length: int, input_variant: str = "fusion", frequency: str = "monthly") -> Path:
    return official_model_dir(timemixer_model_name(input_variant), window_length, frequency)


def timemixer_scratch_dir(window_length: int, input_variant: str = "fusion", frequency: str = "monthly") -> Path:
    return scratch_model_dir(timemixer_model_name(input_variant), window_length, frequency)
