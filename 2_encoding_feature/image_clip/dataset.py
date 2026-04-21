from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGE_MANIFEST_PATH, index_column_name, validate_frequency
from project_shared.io_utils import read_jsonl_records


class ImageDataset:
    def __init__(self, frequency: str = "monthly", manifest_path: str | None = None):
        self.frequency = validate_frequency(frequency)
        self.manifest_path = manifest_path or IMAGE_MANIFEST_PATH
        self.index_column = index_column_name(self.frequency)
        self.project_root = Path(__file__).resolve().parents[2]

    def load_data(self):
        print(f"Loading image data from: {self.manifest_path}")
        data = read_jsonl_records(self.manifest_path)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} image entries")
        return df

    def process_data(self, df):
        df = self._parse_index(df)
        df = self._build_image_path(df)
        df = self._filter_existing_images(df)
        return df

    def _parse_index(self, df):
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df.get("year_month"), format="%Y-%m", errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if self.frequency == "daily":
            df[self.index_column] = df["timestamp"].dt.strftime("%Y-%m-%d")
        else:
            if "year_month" in df.columns:
                df[self.index_column] = df["year_month"].fillna(df["timestamp"].dt.strftime("%Y-%m"))
            else:
                df[self.index_column] = df["timestamp"].dt.strftime("%Y-%m")
        return df

    def _build_image_path(self, df):
        def get_image_path(row):
            for key in ("thumbnail_path", "local_path"):
                relative = row.get(key, "")
                if not relative:
                    continue
                candidate = self.project_root / "1_data_handling" / str(relative)
                if candidate.exists():
                    return str(candidate)
            return ""

        df["image_path"] = df.apply(get_image_path, axis=1)
        return df

    def _filter_existing_images(self, df):
        df["image_exists"] = df["image_path"].apply(lambda value: os.path.exists(value))
        existing_count = int(df["image_exists"].sum())
        missing_count = int(len(df) - existing_count)
        print(f"Found {existing_count} existing images, {missing_count} missing images")
        return df[df["image_exists"]].reset_index(drop=True)

    def get_frequency_groups(self, df):
        return df.groupby(self.index_column)
