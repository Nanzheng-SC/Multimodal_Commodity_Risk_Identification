from __future__ import annotations

from typing import Iterable, List


SUPPORTED_FREQUENCIES = ("daily", "monthly")
SUPPORTED_FREQUENCY_CHOICES = ("daily", "monthly", "both")

WINDOW_LENGTHS_BY_FREQUENCY = {
    "daily": [14, 30, 90],
    "monthly": [3, 6, 12],
}

DEFAULT_WINDOW_LENGTH_BY_FREQUENCY = {
    "daily": 30,
    "monthly": 6,
}

INDEX_COLUMN_BY_FREQUENCY = {
    "daily": "date",
    "monthly": "month",
}

INDEX_LABEL_BY_FREQUENCY = {
    "daily": "date",
    "monthly": "month",
}


def normalize_frequency(frequency: str) -> str:
    value = str(frequency or "").strip().lower()
    if value not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"Unsupported frequency: {frequency}")
    return value


def expand_frequency_choice(frequency: str) -> List[str]:
    value = str(frequency or "").strip().lower()
    if value == "both":
        return list(SUPPORTED_FREQUENCIES)
    return [normalize_frequency(value)]


def index_column_for_frequency(frequency: str) -> str:
    return INDEX_COLUMN_BY_FREQUENCY[normalize_frequency(frequency)]


def default_window_lengths(frequency: str) -> List[int]:
    return list(WINDOW_LENGTHS_BY_FREQUENCY[normalize_frequency(frequency)])


def default_window_length(frequency: str) -> int:
    return DEFAULT_WINDOW_LENGTH_BY_FREQUENCY[normalize_frequency(frequency)]


def all_window_lengths() -> Iterable[int]:
    for frequency in SUPPORTED_FREQUENCIES:
        for window_length in WINDOW_LENGTHS_BY_FREQUENCY[frequency]:
            yield window_length


def is_daily_frequency(frequency: str) -> bool:
    return normalize_frequency(frequency) == "daily"


def is_monthly_frequency(frequency: str) -> bool:
    return normalize_frequency(frequency) == "monthly"


def index_format_hint(frequency: str) -> str:
    frequency = normalize_frequency(frequency)
    return "%Y-%m-%d" if frequency == "daily" else "%Y-%m"
