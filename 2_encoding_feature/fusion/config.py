from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, SEED, UNIFIED_DIM, features_dir, validate_frequency


STAGE_PRIORITY = {"early": 0, "intermediate": 1, "late": 2}

STAGE_METHODS = {
    "early": ["concat", "weighted_sum_fixed", "weighted_sum_learnable"],
    "intermediate": ["gated_fusion", "temporal_attention_fusion"],
    "late": [
        "gru_concat",
        "gru_weighted",
        "gru_gate",
        "gru_gate_baseline",
        "gru_gate_context",
        "gru_gate_context_image_biased",
        "gru_gate_context_image_biased_regularized",
        "gru_gate_concat_preserving",
    ],
}

CANDIDATE_METHODS = [
    ("early", "concat"),
    ("early", "weighted_sum_fixed"),
    ("early", "weighted_sum_learnable"),
    ("intermediate", "gated_fusion"),
    ("intermediate", "temporal_attention_fusion"),
    ("late", "gru_concat"),
    ("late", "gru_weighted"),
    ("late", "gru_gate"),
]

FUSION_CONFIG = {
    "unified_dim": UNIFIED_DIM,
    "device": DEVICE,
    "seed": SEED,
    "default_stage": "late",
    "default_method": "gru_gate",
    "official_stage": "late",
    "official_method": "gru_gate",
    "force_official_canonical_only": False,
    "default_alpha": 0.5,
    "selector_backend": "timemixer",
    "selection_rule": "stable_score",
    "stability_lambda": 0.25,
    "target_mode": "level",
    "forecast_horizon_days": 7,
    "selector_seed_list": [SEED, SEED + 1, SEED + 2],
    "selector_seed_list_final": [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4],
    "attention_heads": 4,
    "dropout": 0.1,
    "gru_layers": 1,
    "gate_mode": "context_mlp",
    "gate_bias_init": -1.0,
    "gate_regularization_lambda": 0.02,
    "gate_max_text_share": 0.45,
    "gate_residual_scale": 0.15,
    "text_modal_dropout": 0.20,
    "image_modal_dropout": 0.05,
    "late_gate_variants": {
        "gru_gate": {
            "gate_mode": "context_mlp",
            "gate_bias_init": -1.0,
            "gate_regularization_lambda": 0.02,
            "gate_max_text_share": 0.45,
            "gate_residual_scale": 0.15,
            "text_modal_dropout": 0.20,
            "image_modal_dropout": 0.05,
        },
        "gru_gate_baseline": {
            "gate_mode": "simple",
            "gate_bias_init": 0.0,
            "gate_regularization_lambda": 0.0,
            "gate_max_text_share": 1.0,
            "gate_residual_scale": 0.0,
            "text_modal_dropout": 0.0,
            "image_modal_dropout": 0.0,
        },
        "gru_gate_context": {
            "gate_mode": "context_mlp",
            "gate_bias_init": 0.0,
            "gate_regularization_lambda": 0.0,
            "gate_max_text_share": 1.0,
            "gate_residual_scale": 0.0,
            "text_modal_dropout": 0.0,
            "image_modal_dropout": 0.0,
        },
        "gru_gate_context_image_biased": {
            "gate_mode": "context_mlp",
            "gate_bias_init": -1.0,
            "gate_regularization_lambda": 0.0,
            "gate_max_text_share": 1.0,
            "gate_residual_scale": 0.10,
            "text_modal_dropout": 0.20,
            "image_modal_dropout": 0.05,
        },
        "gru_gate_context_image_biased_regularized": {
            "gate_mode": "context_mlp",
            "gate_bias_init": -1.0,
            "gate_regularization_lambda": 0.02,
            "gate_max_text_share": 0.45,
            "gate_residual_scale": 0.15,
            "concat_preserve_scale": 0.0,
            "text_modal_dropout": 0.20,
            "image_modal_dropout": 0.05,
        },
        "gru_gate_concat_preserving": {
            "gate_mode": "context_mlp",
            "gate_bias_init": -0.5,
            "gate_regularization_lambda": 0.01,
            "gate_max_text_share": 0.55,
            "gate_residual_scale": 0.10,
            "concat_preserve_scale": 0.35,
            "text_modal_dropout": 0.10,
            "image_modal_dropout": 0.05,
        },
    },
    "selector_window_length_daily": 30,
    "selector_window_length_monthly": 6,
    "selector_split_strategy": "rolling_origin",
    "selector_fixed_val_size_daily": 60,
    "selector_fixed_test_size_daily": 60,
    "selector_fixed_val_size_monthly": 6,
    "selector_fixed_test_size_monthly": 6,
    "selector_rolling_folds": 3,
    "selector_batch_size": 16,
    "selector_max_epochs": 150,
    "selector_patience": 20,
    "selector_learning_rate": 1e-3,
    "selector_weight_decay": 1e-4,
    "selector_hidden_dim": 256,
    "timemixer_batch_size": 16,
    "timemixer_max_epochs": 40,
    "timemixer_patience": 8,
    "timemixer_learning_rate": 5e-4,
    "timemixer_weight_decay": 1e-4,
    "timemixer_d_model": 64,
    "timemixer_n_heads": 4,
    "timemixer_e_layers": 2,
    "timemixer_d_ff": 128,
    "timemixer_dropout": 0.1,
    "timemixer_down_sampling_layers": 2,
    "timemixer_down_sampling_window": 2,
    "timemixer_down_sampling_method": "avg",
    "timemixer_decomp_method": "moving_avg",
    "timemixer_moving_avg": 3,
    "timemixer_top_k": 3,
    "timemixer_use_norm": 1,
    "timemixer_use_future_temporal_feature": 0,
    "timemixer_channel_independence": 0,
    "timemixer_embed": "timeF",
    "timemixer_freq_daily": "d",
    "timemixer_freq_monthly": "m",
    "stage_methods": STAGE_METHODS,
    "candidate_methods": CANDIDATE_METHODS,
    "research_mainline_preferred_stage": "late",
}


def selector_window_length(frequency: str = "monthly") -> int:
    frequency = validate_frequency(frequency)
    return FUSION_CONFIG[f"selector_window_length_{frequency}"]


def split_sizes(frequency: str = "monthly") -> tuple[int, int]:
    frequency = validate_frequency(frequency)
    return (
        FUSION_CONFIG[f"selector_fixed_val_size_{frequency}"],
        FUSION_CONFIG[f"selector_fixed_test_size_{frequency}"],
    )


def timemixer_freq(frequency: str = "monthly") -> str:
    frequency = validate_frequency(frequency)
    return FUSION_CONFIG[f"timemixer_freq_{frequency}"]


def input_paths(frequency: str = "monthly") -> dict:
    frequency = validate_frequency(frequency)
    return {
        "structured": {
            "features": f"{features_dir('structured', frequency)}/structured_features.npy",
            "labels": f"{features_dir('structured', frequency)}/structured_labels.npy",
            "months": f"{features_dir('structured', frequency)}/structured_index.npy",
            "reference": f"{features_dir('structured', frequency)}/structured_reference.npy",
            "manifest": f"{features_dir('structured', frequency)}/structured_feature_manifest.json",
        },
        "text": {
            "embeddings": f"{features_dir('text', frequency)}/text_embeddings.npy",
            "months": f"{features_dir('text', frequency)}/text_index.npy",
            "counts": f"{features_dir('text', frequency)}/text_counts.npy",
            "missing_flags": f"{features_dir('text', frequency)}/text_missing_flags.npy",
        },
        "image": {
            "embeddings": f"{features_dir('image', frequency)}/image_embeddings.npy",
            "months": f"{features_dir('image', frequency)}/image_index.npy",
            "counts": f"{features_dir('image', frequency)}/image_counts.npy",
            "missing_flags": f"{features_dir('image', frequency)}/image_missing_flags.npy",
        },
    }


def output_config(frequency: str = "monthly") -> dict:
    frequency = validate_frequency(frequency)
    return {
        "features_dir": features_dir("fusion", frequency),
        "processed_file": f"fusion_features_{frequency}.csv",
        "features_file": "fusion_features.npy",
        "labels_file": "fusion_labels.npy",
        "months_file": "fusion_index.npy",
        "missing_flags_file": "fusion_missing_flags.npy",
        "scaler_file": "scaler_params.npz",
        "selection_csv": "fusion_selection_results.csv",
        "best_choice_json": "best_fusion_choice.json",
        "method_runs_dir": "method_runs",
        "metrics_file": "metrics.json",
    }


# Backward-compatible monthly aliases
INPUT_PATHS = input_paths("monthly")
OUTPUT_CONFIG = output_config("monthly")
