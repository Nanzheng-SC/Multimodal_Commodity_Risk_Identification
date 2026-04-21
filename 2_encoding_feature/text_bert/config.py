from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, BERT_MODEL_NAME, DEVICE, MAX_TEXT_LENGTH, TEXT_SUMMARY_LENGTH, features_dir


TEXT_ENCODER_CONFIG = {
    "model_name": BERT_MODEL_NAME,
    "fallback_model_names": ["bert-base-uncased"],
    "max_length": MAX_TEXT_LENGTH,
    "batch_size": BATCH_SIZE,
    "device": DEVICE,
}

TEXT_PROCESSING_CONFIG = {
    "summary_length": TEXT_SUMMARY_LENGTH,
}


def text_output_config(frequency: str = "monthly") -> dict:
    return {
        "features_dir": features_dir("text", frequency),
        "processed_file": f"text_features_{frequency}.csv",
        "embeddings_file": "text_embeddings.npy",
        "months_file": "text_index.npy",
        "counts_file": "text_counts.npy",
        "missing_flags_file": "text_missing_flags.npy",
    }


TEXT_OUTPUT_CONFIG = text_output_config("monthly")
