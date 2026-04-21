from __future__ import annotations

from pathlib import Path

from project_shared.frequency import normalize_frequency


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_HANDLING_ROOT = PROJECT_ROOT / "1_data_handling"
RAW_ROOT = DATA_HANDLING_ROOT / "raw"
STRUCTURED_RAW_ROOT = RAW_ROOT / "structured"
TEXT_RAW_ROOT = RAW_ROOT / "text"
IMAGE_RAW_ROOT = RAW_ROOT / "image"
EVENT_RAW_ROOT = RAW_ROOT / "event"

STRUCTURED_DAILY_PATH = STRUCTURED_RAW_ROOT / "structured_daily_merged.csv"
STRUCTURED_MONTHLY_DERIVED_PATH = STRUCTURED_RAW_ROOT / "structured_monthly_derived.csv"
STRUCTURED_MANIFEST_PATH = STRUCTURED_RAW_ROOT / "structured_dataset_manifest.json"

TEXT_DOCUMENTS_CLEANED_GZ_PATH = TEXT_RAW_ROOT / "text_documents_multisource_cleaned.jsonl.gz"
TEXT_DAILY_COVERAGE_JSON_PATH = TEXT_RAW_ROOT / "text_daily_coverage.json"
TEXT_DAILY_COVERAGE_CSV_PATH = TEXT_RAW_ROOT / "text_daily_coverage.csv"

IMAGE_MANIFEST_CLEANED_GZ_PATH = IMAGE_RAW_ROOT / "image_manifest_commons_cleaned.jsonl.gz"
IMAGE_THUMBNAIL_DIR = IMAGE_RAW_ROOT / "thumbnails_webp"
IMAGE_DAILY_COVERAGE_JSON_PATH = IMAGE_RAW_ROOT / "image_daily_coverage.json"
IMAGE_DAILY_COVERAGE_CSV_PATH = IMAGE_RAW_ROOT / "image_daily_coverage.csv"
IMAGE_COLLECTION_REPORT_JSON_PATH = IMAGE_RAW_ROOT / "image_collection_report.json"

EVENT_MANIFEST_PATH = EVENT_RAW_ROOT / "event_manifest.jsonl"

DATA_INVENTORY_JSON_PATH = DATA_HANDLING_ROOT / "DATA_INVENTORY.json"
DATA_INVENTORY_MD_PATH = DATA_HANDLING_ROOT / "DATA_INVENTORY.md"

ENCODING_ROOT = PROJECT_ROOT / "2_encoding_feature"
ENCODING_OUTPUT_ROOT = ENCODING_ROOT / "outputs"

MODELING_ROOT = PROJECT_ROOT / "3_modeling"
MODELING_RESULTS_ROOT = MODELING_ROOT / "results"
MODELING_OFFICIAL_ROOT = MODELING_RESULTS_ROOT / "official"
MODELING_COMPARISON_ROOT = MODELING_RESULTS_ROOT / "comparison"
MODELING_SCRATCH_ROOT = MODELING_RESULTS_ROOT / "_scratch"

DASHBOARD_ROOT = PROJECT_ROOT / "4_decision_rl"
DASHBOARD_JSON_PATH = DASHBOARD_ROOT / "decision_dashboard_data.json"
DASHBOARD_JS_PATH = DASHBOARD_ROOT / "decision_dashboard_data.js"


def structured_source_path(frequency: str) -> Path:
    frequency = normalize_frequency(frequency)
    return STRUCTURED_DAILY_PATH if frequency == "daily" else STRUCTURED_MONTHLY_DERIVED_PATH


def preferred_text_input_path() -> Path:
    return TEXT_DOCUMENTS_CLEANED_GZ_PATH


def preferred_image_manifest_path() -> Path:
    return IMAGE_MANIFEST_CLEANED_GZ_PATH


def frequency_output_root(frequency: str) -> Path:
    return ENCODING_OUTPUT_ROOT / normalize_frequency(frequency)


def feature_root(kind: str, frequency: str) -> Path:
    return frequency_output_root(frequency) / f"{kind}_features"


def time_series_root(input_variant: str, frequency: str) -> Path:
    suffix = {
        "fusion": "time_series",
        "text": "time_series_text",
        "image": "time_series_image",
        "structured": "time_series_structured",
    }[input_variant]
    return frequency_output_root(frequency) / suffix


def timemixer_model_name(input_variant: str = "fusion") -> str:
    return {
        "fusion": "timemixer",
        "text": "timemixer_text",
        "image": "timemixer_image",
        "structured": "timemixer_structured",
    }[input_variant]


def official_model_dir(model_name: str, window_length: int, frequency: str) -> Path:
    return MODELING_OFFICIAL_ROOT / normalize_frequency(frequency) / model_name / f"window_{window_length}"


def scratch_model_dir(model_name: str, window_length: int, frequency: str) -> Path:
    return MODELING_SCRATCH_ROOT / normalize_frequency(frequency) / model_name / f"window_{window_length}"


def timemixer_official_dir(window_length: int, input_variant: str = "fusion", frequency: str = "monthly") -> Path:
    return official_model_dir(timemixer_model_name(input_variant), window_length, frequency)


def timemixer_scratch_dir(window_length: int, input_variant: str = "fusion", frequency: str = "monthly") -> Path:
    return scratch_model_dir(timemixer_model_name(input_variant), window_length, frequency)
