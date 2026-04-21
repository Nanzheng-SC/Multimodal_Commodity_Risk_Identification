#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


ENCODING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ENCODING_ROOT)
DATA_HANDLING_ROOT = os.path.join(PROJECT_ROOT, "1_data_handling")
for path in (ENCODING_ROOT, PROJECT_ROOT, DATA_HANDLING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import STRUCTURED_DROP_COLUMNS, features_dir, index_column_name, structured_data_path, validate_frequency
from raw.structured.field_registry import modeling_field_names
from project_shared.targets import (
    PRIMARY_TARGET_COLUMN,
    REFERENCE_ALIAS_COLUMN,
    REFERENCE_PRICE_COLUMN,
    STEP_RETURN_COLUMN,
    STEP_VOLATILITY_COLUMN,
    build_price_targets,
)


REGIME_DERIVED_COLUMNS = [
    "brent_return_3d",
    "brent_return_7d",
    "brent_return_14d",
    "brent_return_30d",
    "brent_volatility_14d",
    "brent_volatility_30d",
    "wti_brent_spread",
    "usd_index_return_7d",
    "gpr_change_7d",
    "epu_change_7d",
    "brent_zscore_30d",
    "high_volatility_regime",
    "fast_uptrend_regime",
]


def _output_paths(frequency: str) -> dict[str, str]:
    root = features_dir("structured", frequency)
    os.makedirs(root, exist_ok=True)
    return {
        "root": root,
        "processed": os.path.join(root, f"structured_{frequency}_processed.csv"),
        "features": os.path.join(root, "structured_features.npy"),
        "labels": os.path.join(root, "structured_labels.npy"),
        "index": os.path.join(root, "structured_index.npy"),
        "index_legacy": os.path.join(root, "structured_months.npy"),
        "reference": os.path.join(root, "structured_reference.npy"),
        "manifest": os.path.join(root, "structured_feature_manifest.json"),
    }


def _load_source_frame(frequency: str) -> pd.DataFrame:
    path = structured_data_path(frequency)
    df = pd.read_csv(path, encoding="utf-8-sig")
    index_column = index_column_name(frequency)
    if frequency == "daily":
        if "date" not in df.columns:
            raise ValueError(f"Daily structured file must contain date column: {path}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df[index_column] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        if "month" not in df.columns:
            if "date" in df.columns:
                df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
            else:
                raise ValueError(f"Monthly structured file must contain month column: {path}")
        df["month"] = df["month"].astype(str)
        df = df.sort_values("month").reset_index(drop=True)
        df[index_column] = df["month"]

    if PRIMARY_TARGET_COLUMN not in df.columns or REFERENCE_ALIAS_COLUMN not in df.columns:
        df = build_price_targets(df, price_column=REFERENCE_PRICE_COLUMN)

    if STEP_RETURN_COLUMN not in df.columns:
        df[STEP_RETURN_COLUMN] = pd.to_numeric(df[REFERENCE_PRICE_COLUMN], errors="coerce").pct_change()
    if STEP_VOLATILITY_COLUMN not in df.columns:
        df[STEP_VOLATILITY_COLUMN] = pd.to_numeric(df[STEP_RETURN_COLUMN], errors="coerce").rolling(window=7, min_periods=2).std()

    return df


def _add_regime_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical-only state features for daily Brent modeling."""
    if REFERENCE_PRICE_COLUMN not in df.columns:
        return df

    brent = pd.to_numeric(df[REFERENCE_PRICE_COLUMN], errors="coerce").ffill()
    returns_1d = brent.pct_change()

    for window in (3, 7, 14, 30):
        df[f"brent_return_{window}d"] = brent.pct_change(periods=window)

    df["brent_volatility_14d"] = returns_1d.rolling(window=14, min_periods=3).std()
    df["brent_volatility_30d"] = returns_1d.rolling(window=30, min_periods=5).std()

    if "WTI" in df.columns:
        df["wti_brent_spread"] = pd.to_numeric(df["WTI"], errors="coerce").ffill() - brent
    if "USD_Index" in df.columns:
        df["usd_index_return_7d"] = pd.to_numeric(df["USD_Index"], errors="coerce").ffill().pct_change(periods=7)
    if "GPR" in df.columns:
        df["gpr_change_7d"] = pd.to_numeric(df["GPR"], errors="coerce").ffill().diff(periods=7)
    if "EPU" in df.columns:
        df["epu_change_7d"] = pd.to_numeric(df["EPU"], errors="coerce").ffill().diff(periods=7)

    rolling_mean = brent.rolling(window=30, min_periods=5).mean()
    rolling_std = brent.rolling(window=30, min_periods=5).std()
    df["brent_zscore_30d"] = (brent - rolling_mean) / rolling_std.replace(0, np.nan)

    vol_threshold = df["brent_volatility_30d"].rolling(window=252, min_periods=30).quantile(0.75)
    uptrend_threshold = df["brent_return_14d"].rolling(window=252, min_periods=30).quantile(0.75)
    df["high_volatility_regime"] = (df["brent_volatility_30d"] > vol_threshold).astype(float)
    df["fast_uptrend_regime"] = (df["brent_return_14d"] > uptrend_threshold).astype(float)

    for column in REGIME_DERIVED_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return df


def process_structured_data(frequency: str = "monthly"):
    frequency = validate_frequency(frequency)
    index_column = index_column_name(frequency)
    paths = _output_paths(frequency)

    print(f"Reading structured data for frequency={frequency} from: {structured_data_path(frequency)}")
    df = _load_source_frame(frequency)
    if frequency == "daily":
        df = _add_regime_aware_features(df)
    input_samples = len(df)

    dropped_columns = [column for column in STRUCTURED_DROP_COLUMNS if column in df.columns]
    if dropped_columns:
        df = df.drop(columns=dropped_columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    required_numeric = {REFERENCE_ALIAS_COLUMN, PRIMARY_TARGET_COLUMN}
    required_numeric.update({STEP_RETURN_COLUMN, STEP_VOLATILITY_COLUMN})
    missing_numeric = [column for column in required_numeric if column not in numeric_cols]
    if missing_numeric:
        raise ValueError(f"Missing required numeric columns after loading structured data: {missing_numeric}")

    modeled_columns = set(modeling_field_names())
    feature_cols = [column for column in df.columns if column in modeled_columns and column in numeric_cols]
    if not feature_cols:
        raise ValueError("No active structured modeling fields were found in the loaded structured dataset.")

    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].mean())

    labels = pd.to_numeric(df[PRIMARY_TARGET_COLUMN], errors="coerce")
    reference = pd.to_numeric(df[REFERENCE_ALIAS_COLUMN], errors="coerce")
    valid_samples = int(labels.notna().sum())
    missing_samples = int(input_samples - valid_samples)
    min_index = str(df[index_column].min()) if input_samples else "N/A"
    max_index = str(df[index_column].max()) if input_samples else "N/A"

    output_cols = [index_column, *feature_cols, REFERENCE_ALIAS_COLUMN, PRIMARY_TARGET_COLUMN]
    if "target_brent_day7" in df.columns:
        output_cols.append("target_brent_day7")
    df[output_cols].to_csv(paths["processed"], index=False, encoding="utf-8")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    label_values = labels.to_numpy(dtype=np.float32)
    index_values = df[index_column].to_numpy()
    reference_values = reference.to_numpy(dtype=np.float32)

    np.save(paths["features"], features)
    np.save(paths["labels"], label_values)
    np.save(paths["index"], index_values)
    np.save(paths["index_legacy"], index_values)
    np.save(paths["reference"], reference_values)

    manifest = {
        "frequency": frequency,
        "index_column": index_column,
        "dropped_columns": dropped_columns,
        "feature_columns": feature_cols,
        "feature_dim": int(features.shape[1]),
        "label_column": PRIMARY_TARGET_COLUMN,
        "reference_column": REFERENCE_ALIAS_COLUMN,
        "index_min": min_index,
        "index_max": max_index,
        "derived_context_columns": [
            STEP_RETURN_COLUMN,
            STEP_VOLATILITY_COLUMN,
            *[column for column in REGIME_DERIVED_COLUMNS if column in feature_cols],
        ],
    }
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("=== Structured data summary ===")
    print(f"Frequency: {frequency}")
    print(f"Input samples: {input_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"Missing label samples: {missing_samples}")
    print(f"Index coverage: {min_index} to {max_index}")
    print(f"Dropped columns: {dropped_columns if dropped_columns else 'None'}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {label_values.shape}")
    print(f"Saved CSV: {paths['processed']}")
    print(f"Saved manifest: {paths['manifest']}")
    print("===============================")
    return df[output_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process structured daily/monthly Brent datasets.")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_structured_data(frequency=args.frequency)
