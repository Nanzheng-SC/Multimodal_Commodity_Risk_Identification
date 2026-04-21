from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from fusion.config import input_paths, output_config
from project_shared.frequency import index_column_for_frequency


class AlignAndMerge:
    def __init__(self, frequency: str = "monthly", output_dir: str | Path | None = None) -> None:
        self.frequency = frequency
        self.index_column = index_column_for_frequency(frequency)
        config = output_config(frequency)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(config["features_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path_config = input_paths(frequency)

    def load_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        structured_manifest_path = Path(self.path_config["structured"]["manifest"])
        feature_names = []
        if structured_manifest_path.exists():
            feature_names = json.loads(structured_manifest_path.read_text(encoding="utf-8")).get("feature_columns", [])
        return {
            "structured": {
                "features": np.load(self.path_config["structured"]["features"], allow_pickle=True),
                "labels": np.load(self.path_config["structured"]["labels"], allow_pickle=True),
                "months": np.load(self.path_config["structured"]["months"], allow_pickle=True),
                "reference": np.load(self.path_config["structured"]["reference"], allow_pickle=True),
                "feature_names": feature_names,
            },
            "text": {
                "embeddings": np.load(self.path_config["text"]["embeddings"], allow_pickle=True),
                "months": np.load(self.path_config["text"]["months"], allow_pickle=True),
                "counts": np.load(self.path_config["text"]["counts"], allow_pickle=True),
                "missing_flags": np.load(self.path_config["text"]["missing_flags"], allow_pickle=True),
            },
            "image": {
                "embeddings": np.load(self.path_config["image"]["embeddings"], allow_pickle=True),
                "months": np.load(self.path_config["image"]["months"], allow_pickle=True),
                "counts": np.load(self.path_config["image"]["counts"], allow_pickle=True),
                "missing_flags": np.load(self.path_config["image"]["missing_flags"], allow_pickle=True),
            },
        }

    def align_data(self, data: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        structured_df = pd.DataFrame(
            {
                self.index_column: data["structured"]["months"],
                "structured_features": list(data["structured"]["features"]),
                "label": data["structured"]["labels"],
                "reference": data["structured"]["reference"],
            }
        )
        text_df = pd.DataFrame(
            {
                self.index_column: data["text"]["months"],
                "text_embedding": list(data["text"]["embeddings"]),
                "text_count": data["text"]["counts"],
                "text_missing_flag": data["text"]["missing_flags"],
            }
        )
        image_df = pd.DataFrame(
            {
                self.index_column: data["image"]["months"],
                "image_embedding": list(data["image"]["embeddings"]),
                "image_count": data["image"]["counts"],
                "image_missing_flag": data["image"]["missing_flags"],
            }
        )
        merged_df = structured_df.merge(text_df, on=self.index_column, how="left")
        merged_df = merged_df.merge(image_df, on=self.index_column, how="left")
        merged_df.attrs["structured_feature_names"] = data["structured"].get("feature_names", [])
        return self._handle_missing_values(merged_df)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text_embedding" in df.columns:
            text_dim = len(df["text_embedding"].dropna().iloc[0]) if not df["text_embedding"].dropna().empty else 768
            df["text_embedding"] = df["text_embedding"].apply(
                lambda value: value if isinstance(value, np.ndarray) else np.zeros(text_dim)
            )
            df["text_count"] = df["text_count"].fillna(0).astype(int)
            df["text_missing_flag"] = df["text_missing_flag"].fillna(1).astype(int)

        if "image_embedding" in df.columns:
            image_dim = len(df["image_embedding"].dropna().iloc[0]) if not df["image_embedding"].dropna().empty else 512
            df["image_embedding"] = df["image_embedding"].apply(
                lambda value: value if isinstance(value, np.ndarray) else np.zeros(image_dim)
            )
            df["image_count"] = df["image_count"].fillna(0).astype(int)
            df["image_missing_flag"] = df["image_missing_flag"].fillna(1).astype(int)

        return df

    def get_aligned_arrays(self, merged_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "structured_features": np.array(merged_df["structured_features"].tolist(), dtype=np.float32),
            "text_embeddings": np.array(merged_df["text_embedding"].tolist(), dtype=np.float32),
            "image_embeddings": np.array(merged_df["image_embedding"].tolist(), dtype=np.float32),
            "labels": merged_df["label"].to_numpy(dtype=np.float32),
            "reference": merged_df["reference"].to_numpy(dtype=np.float32),
            "months": merged_df[self.index_column].to_numpy(),
            "text_counts": merged_df["text_count"].to_numpy(dtype=np.int64),
            "image_counts": merged_df["image_count"].to_numpy(dtype=np.int64),
            "missing_flags": merged_df[["text_missing_flag", "image_missing_flag"]].to_numpy(dtype=np.int64),
            "structured_feature_names": merged_df.attrs.get("structured_feature_names", []),
        }
