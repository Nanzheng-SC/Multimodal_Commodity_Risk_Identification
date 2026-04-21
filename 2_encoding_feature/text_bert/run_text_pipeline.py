from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import features_dir, index_column_name, validate_frequency
from text_bert.config import text_output_config
from text_bert.dataset import TextDataset
from text_bert.encoder_local import TextEncoder
from text_bert.pooling import TextPooling


def _structured_index_path(frequency: str) -> str:
    return os.path.join(features_dir("structured", frequency), "structured_index.npy")


def _load_structured_index(frequency: str):
    path = _structured_index_path(frequency)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    print(f"Warning: structured index file not found for frequency={frequency}. Using text-only index.")
    return None


def _derive_monthly_from_daily_if_available(output_dir: str) -> pd.DataFrame | None:
    daily_dir = features_dir("text", "daily")
    daily_embeddings = os.path.join(daily_dir, "text_embeddings.npy")
    daily_index = os.path.join(daily_dir, "text_index.npy")
    daily_counts = os.path.join(daily_dir, "text_counts.npy")
    daily_flags = os.path.join(daily_dir, "text_missing_flags.npy")
    if not all(os.path.exists(path) for path in (daily_embeddings, daily_index, daily_counts, daily_flags)):
        return None
    embeddings = np.load(daily_embeddings, allow_pickle=True)
    dates = np.load(daily_index, allow_pickle=True).astype(str)
    counts = np.load(daily_counts, allow_pickle=True).astype(np.int64)
    flags = np.load(daily_flags, allow_pickle=True).astype(np.int64)
    frame = pd.DataFrame(
        {
            "date": dates,
            "embedding": list(embeddings),
            "text_count": counts,
            "text_missing_flag": flags,
        }
    )
    frame["month"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m")
    monthly_rows = []
    for month, group in frame.groupby("month"):
        valid = group[group["text_count"] > 0]
        if valid.empty:
            monthly_rows.append({"month": month, "embedding": np.zeros_like(frame.iloc[0]["embedding"]), "text_count": 0, "text_missing_flag": 1})
            continue
        weight = valid["text_count"].to_numpy(dtype=np.float32)
        stacked = np.stack(valid["embedding"].to_list()).astype(np.float32)
        monthly_rows.append(
            {
                "month": month,
                "embedding": np.average(stacked, axis=0, weights=weight),
                "text_count": int(valid["text_count"].sum()),
                "text_missing_flag": 0,
            }
        )
    return pd.DataFrame(monthly_rows)


def run_text_pipeline(frequency: str = "monthly"):
    frequency = validate_frequency(frequency)
    index_column = index_column_name(frequency)
    structured_index = _load_structured_index(frequency)
    output_config = text_output_config(frequency)
    output_dir = output_config["features_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if frequency == "monthly":
        derived = _derive_monthly_from_daily_if_available(output_dir)
    else:
        derived = None

    if derived is not None:
        results_df = derived.copy()
        input_samples = int(results_df["text_count"].sum())
        valid_samples = int((results_df["text_count"] > 0).sum())
        missing_samples = int(len(results_df) - valid_samples)
    else:
        dataset = TextDataset(frequency=frequency)
        encoder = TextEncoder()
        pooling = TextPooling()
        df = dataset.load_data()
        input_samples = len(df)
        df = dataset.process_data(df)
        valid_samples = len(df)
        missing_samples = input_samples - valid_samples
        grouped = dataset.get_frequency_groups(df)
        results_df = pd.DataFrame(pooling.process_all_periods(grouped, encoder, index_column=index_column))

    if structured_index is not None:
        all_index = pd.DataFrame({index_column: structured_index})
        results_df = pd.merge(all_index, results_df, on=index_column, how="left")
        embedding_dim = results_df["embedding"].apply(lambda value: len(value) if isinstance(value, np.ndarray) else 0).max()
        if not embedding_dim:
            embedding_dim = 768
        results_df["embedding"] = results_df["embedding"].apply(
            lambda value: value if isinstance(value, np.ndarray) else np.zeros(embedding_dim, dtype=np.float32)
        )
        results_df["text_count"] = results_df["text_count"].fillna(0).astype(int)
        results_df["text_missing_flag"] = results_df["text_missing_flag"].fillna(1).astype(int)

    min_index = results_df[index_column].min() if not results_df.empty else "N/A"
    max_index = results_df[index_column].max() if not results_df.empty else "N/A"

    embeddings = np.array(results_df["embedding"].tolist(), dtype=np.float32)
    index_values = results_df[index_column].values
    counts = results_df["text_count"].values
    missing_flags = results_df["text_missing_flag"].values

    processed_file = os.path.join(output_dir, output_config["processed_file"])
    embeddings_file = os.path.join(output_dir, output_config["embeddings_file"])
    index_file = os.path.join(output_dir, output_config["months_file"])
    legacy_index_file = os.path.join(output_dir, "text_months.npy")
    counts_file = os.path.join(output_dir, output_config["counts_file"])
    missing_flags_file = os.path.join(output_dir, output_config["missing_flags_file"])

    results_df.to_csv(processed_file, index=False, encoding="utf-8")
    np.save(embeddings_file, embeddings)
    np.save(index_file, index_values)
    np.save(legacy_index_file, index_values)
    np.save(counts_file, counts)
    np.save(missing_flags_file, missing_flags)

    print("=== Text encoding and aggregation summary ===")
    print(f"Frequency: {frequency}")
    print(f"Input samples: {input_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"Missing samples: {missing_samples}")
    print(f"Index coverage: {min_index} to {max_index}")
    print(f"Output shape: {results_df.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {processed_file}")
    print("============================================")
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run the text encoding pipeline.")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_text_pipeline(frequency=args.frequency)
