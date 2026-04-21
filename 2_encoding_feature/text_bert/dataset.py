from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TEXT_DATA_PATH, index_column_name, validate_frequency
from text_bert.config import TEXT_PROCESSING_CONFIG
from project_shared.io_utils import read_jsonl_records


class TextDataset:
    def __init__(self, frequency: str = "monthly", data_path: str | None = None):
        self.frequency = validate_frequency(frequency)
        self.data_path = data_path or TEXT_DATA_PATH
        self.summary_length = TEXT_PROCESSING_CONFIG["summary_length"]
        self.index_column = index_column_name(self.frequency)

    def load_data(self):
        print(f"Loading text data from: {self.data_path}")
        data = read_jsonl_records(self.data_path)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} text documents")
        return df

    def process_data(self, df):
        df = self._parse_index(df)
        df = self._prepare_text_input(df)
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

    def _prepare_text_input(self, df):
        def get_text_input(row):
            title = row.get("title", "")
            summary = row.get("summary", "")
            text = row.get("text", "")
            if summary:
                return f"{title}\n{summary}"
            return f"{title}\n{text[:self.summary_length]}"

        df["text_input"] = df.apply(get_text_input, axis=1)
        df = df[df["text_input"].str.strip() != ""].reset_index(drop=True)
        return df

    def get_frequency_groups(self, df):
        return df.groupby(self.index_column)
