from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.align_and_merge import AlignAndMerge
from fusion.config import FUSION_CONFIG, STAGE_METHODS, output_config, selector_window_length
from fusion.selector import FusionMethodSelector
from project_shared.frequency import normalize_frequency


def _print_summary(method_id: str, result: dict, aligned_arrays: dict, selector_backend: str, frequency: str) -> None:
    labels = aligned_arrays["labels"]
    index_values = aligned_arrays["months"]
    valid_samples = int(np.sum(~np.isnan(labels)))
    print("=== Multi-modal fusion summary ===")
    print(f"Frequency: {frequency}")
    print(f"Method: {method_id}")
    print(f"Input samples: {len(index_values)}")
    print(f"Valid samples: {valid_samples}")
    print(f"Missing label samples: {len(index_values) - valid_samples}")
    print(f"Index coverage: {index_values.min()} to {index_values.max()}")
    print(f"Selector backend: {selector_backend}")
    if "selection_rule" in result:
        print(f"Selection rule: {result['selection_rule']}")
    if "stable_score" in result:
        print(f"Stable score: {result['stable_score']:.6f}")
    print(f"Validation RMSE: {result['val_rmse']:.6f}")
    print(f"Test RMSE: {result['test_rmse']:.6f}")
    print(f"Result directory: {result['method_dir']}")
    print("==================================")


def _parse_seed_list(raw_value: str | None) -> list[int] | None:
    if not raw_value:
        return None
    seeds = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or None


def _parse_candidate_methods(raw_value: str | None) -> list[tuple[str, str]] | None:
    if not raw_value:
        return None
    parsed: list[tuple[str, str]] = []
    for item in raw_value.split(","):
        method_id = item.strip()
        if not method_id:
            continue
        if "." not in method_id:
            raise ValueError(f"Candidate method must use stage.method format: {method_id}")
        stage, method = method_id.split(".", 1)
        if stage not in STAGE_METHODS or method not in STAGE_METHODS[stage]:
            raise ValueError(f"Unsupported candidate method: {method_id}")
        parsed.append((stage, method))
    return parsed or None


def run_fusion_pipeline(
    fusion_stage: str | None = None,
    fusion_method: str | None = None,
    select_best: bool = False,
    window_length: int | None = None,
    fused_dim: int | None = None,
    alpha: float | None = None,
    selector_backend: str | None = None,
    frequency: str = "monthly",
    selection_rule: str | None = None,
    stability_lambda: float | None = None,
    seed_list: list[int] | None = None,
    candidate_methods: list[tuple[str, str]] | None = None,
    target_mode: str | None = None,
    forecast_horizon_days: int | None = None,
    gate_mode: str | None = None,
    gate_bias_init: float | None = None,
    gate_regularization_lambda: float | None = None,
    gate_max_text_share: float | None = None,
    gate_residual_scale: float | None = None,
    text_modal_dropout: float | None = None,
    image_modal_dropout: float | None = None,
    sync_canonical: bool = True,
):
    frequency = normalize_frequency(frequency)
    output_paths = output_config(frequency)
    stage = fusion_stage or FUSION_CONFIG["default_stage"]
    if fusion_method is not None:
        method = fusion_method
    elif stage == FUSION_CONFIG["default_stage"]:
        method = FUSION_CONFIG["default_method"]
    else:
        method = STAGE_METHODS[stage][0]
    window_length = window_length or selector_window_length(frequency)
    fused_dim = fused_dim or FUSION_CONFIG["unified_dim"]
    alpha = alpha if alpha is not None else FUSION_CONFIG["default_alpha"]
    selector_backend = selector_backend or FUSION_CONFIG["selector_backend"]
    gate_overrides = {
        "gate_mode": gate_mode,
        "gate_bias_init": gate_bias_init,
        "gate_regularization_lambda": gate_regularization_lambda,
        "gate_max_text_share": gate_max_text_share,
        "gate_residual_scale": gate_residual_scale,
        "text_modal_dropout": text_modal_dropout,
        "image_modal_dropout": image_modal_dropout,
    }
    for key, value in gate_overrides.items():
        if value is not None:
            FUSION_CONFIG[key] = value

    if stage not in STAGE_METHODS:
        raise ValueError(f"Unsupported fusion stage: {stage}")
    if not select_best and method not in STAGE_METHODS[stage]:
        raise ValueError(f"Unsupported fusion method for stage {stage}: {method}")

    aligner = AlignAndMerge(frequency=frequency)
    data = aligner.load_data()
    merged_df = aligner.align_data(data)
    aligned_arrays = aligner.get_aligned_arrays(merged_df)
    selector = FusionMethodSelector(
        aligned_data=aligned_arrays,
        output_dir=Path(output_paths["features_dir"]),
        selector_backend=selector_backend,
        frequency=frequency,
        selection_rule=selection_rule,
        stability_lambda=stability_lambda,
        seed_list=seed_list,
        target_mode=target_mode,
        forecast_horizon_days=forecast_horizon_days,
    )

    if select_best:
        if FUSION_CONFIG.get("force_official_canonical_only", False):
            result = selector.run_single_method(
                stage=FUSION_CONFIG["official_stage"],
                method=FUSION_CONFIG["official_method"],
                window_length=window_length,
                fused_dim=fused_dim,
                alpha=alpha,
                sync_canonical=True,
            )
        else:
            result = selector.select_best(
                window_length=window_length,
                fused_dim=fused_dim,
                alpha=alpha,
                candidates=candidate_methods,
            )
    else:
        result = selector.run_single_method(
            stage=stage,
            method=method,
            window_length=window_length,
            fused_dim=fused_dim,
            alpha=alpha,
            sync_canonical=sync_canonical,
        )

    _print_summary(result["method_id"], result, aligned_arrays, selector_backend, frequency)
    return {
        "aligned_frame": merged_df,
        "best_result": result,
        "canonical_dir": str(Path(output_paths["features_dir"]).resolve()),
        "frequency": frequency,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run dual-frequency text-image-structured fusion and selection.")
    parser.add_argument(
        "--fusion-stage",
        choices=sorted(STAGE_METHODS.keys()),
        default=FUSION_CONFIG["default_stage"],
        help="Fusion stage to run when not selecting the best method.",
    )
    parser.add_argument(
        "--fusion-method",
        default=None,
        help="Fusion method to run within the selected stage.",
    )
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="Run candidate fusion methods and select the best one by the configured validation-only rule.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Sequence window length used in method selection.",
    )
    parser.add_argument(
        "--fused-dim",
        type=int,
        default=FUSION_CONFIG["unified_dim"],
        help="Unified fusion dimension for the modality merge output.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=FUSION_CONFIG["default_alpha"],
        help="Fixed alpha used by weighted-sum fusion methods.",
    )
    parser.add_argument(
        "--selector-backend",
        choices=["sequence_pool", "timemixer"],
        default=FUSION_CONFIG["selector_backend"],
        help="Backend used to score fusion methods.",
    )
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument(
        "--selection-rule",
        choices=["val_rmse", "stable_score"],
        default=FUSION_CONFIG.get("selection_rule", "stable_score"),
        help="Validation-only rule used to rank fusion methods.",
    )
    parser.add_argument(
        "--stability-lambda",
        type=float,
        default=FUSION_CONFIG.get("stability_lambda", 0.25),
        help="Penalty weight for validation RMSE standard deviation when using stable_score.",
    )
    parser.add_argument(
        "--seed-list",
        default=None,
        help="Comma-separated selector seeds, for example 42,43,44.",
    )
    parser.add_argument(
        "--candidate-methods",
        default=None,
        help="Comma-separated stage.method ids to compare, for example late.gru_gate,late.gru_weighted.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["level", "residual", "relative_residual"],
        default=FUSION_CONFIG.get("target_mode", "level"),
    )
    parser.add_argument("--forecast-horizon-days", type=int, default=FUSION_CONFIG.get("forecast_horizon_days", 7))
    parser.add_argument("--gate-mode", choices=["simple", "context_mlp"], default=None)
    parser.add_argument("--gate-bias-init", type=float, default=None)
    parser.add_argument("--gate-regularization-lambda", type=float, default=None)
    parser.add_argument("--gate-max-text-share", type=float, default=None)
    parser.add_argument("--gate-residual-scale", type=float, default=None)
    parser.add_argument("--text-modal-dropout", type=float, default=None)
    parser.add_argument("--image-modal-dropout", type=float, default=None)
    parser.add_argument("--no-sync-canonical", action="store_true", help="Do not overwrite canonical fusion feature files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fusion_pipeline(
        fusion_stage=args.fusion_stage,
        fusion_method=args.fusion_method,
        select_best=args.select_best,
        window_length=args.window_length,
        fused_dim=args.fused_dim,
        alpha=args.alpha,
        selector_backend=args.selector_backend,
        frequency=args.frequency,
        selection_rule=args.selection_rule,
        stability_lambda=args.stability_lambda,
        seed_list=_parse_seed_list(args.seed_list),
        candidate_methods=_parse_candidate_methods(args.candidate_methods),
        target_mode=args.target_mode,
        forecast_horizon_days=args.forecast_horizon_days,
        gate_mode=args.gate_mode,
        gate_bias_init=args.gate_bias_init,
        gate_regularization_lambda=args.gate_regularization_lambda,
        gate_max_text_share=args.gate_max_text_share,
        gate_residual_scale=args.gate_residual_scale,
        text_modal_dropout=args.text_modal_dropout,
        image_modal_dropout=args.image_modal_dropout,
        sync_canonical=not args.no_sync_canonical,
    )
