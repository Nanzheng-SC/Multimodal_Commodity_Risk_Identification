from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, CLIP_MODEL_NAME, DEVICE, IMAGE_SIZE, features_dir


IMAGE_ENCODER_CONFIG = {
    "model_name": CLIP_MODEL_NAME,
    "fallback_model_names": ["openai/clip-vit-base-patch32"],
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "device": DEVICE,
}


def image_output_config(frequency: str = "monthly") -> dict:
    return {
        "features_dir": features_dir("image", frequency),
        "processed_file": f"image_features_{frequency}.csv",
        "embeddings_file": "image_embeddings.npy",
        "months_file": "image_index.npy",
        "counts_file": "image_counts.npy",
        "missing_flags_file": "image_missing_flags.npy",
    }


IMAGE_OUTPUT_CONFIG = image_output_config("monthly")
