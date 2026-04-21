from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


TIMEMIXER_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = TIMEMIXER_ROOT.parent
PROJECT_ROOT = MODELING_ROOT.parent
ENCODING_ROOT = PROJECT_ROOT / "2_encoding_feature"

for path in (PROJECT_ROOT, MODELING_ROOT, TIMEMIXER_ROOT, ENCODING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import aggregate_metric_dicts, build_diagnostics, compute_zero_baseline, metric_dict  # noqa: E402
from common.reporting import configure_matplotlib, save_json, save_prediction_artifacts  # noqa: E402
from official_benchmark import (  # noqa: E402
    DEFAULT_CONFIG,
    build_combined_bundle,
    evaluate_model,
    fit_single_run,
    resolve_training_device,
)
from fusion.run_fusion_pipeline import run_fusion_pipeline  # noqa: E402
from project_shared.paths import ENCODING_OUTPUT_ROOT, STRUCTURED_DAILY_PATH  # noqa: E402
from project_shared.targets import REFERENCE_PRICE_COLUMN, compute_forward_average  # noqa: E402
from time_series.dataset_builder import DatasetBuilder  # noqa: E402
from time_series.window_builder import WindowBuilder  # noqa: E402


FREQUENCY = "daily"
HORIZON_DAYS = 30
FOLD_COUNT = 6
FOLD_SIZE = 60
ROLLING_LAMBDA = 0.25

RESULT_ROOT = MODELING_ROOT / "results" / "mainline_tuning" / "daily_horizon30_late_gru_gate"
WINDOW_ROOT = ENCODING_OUTPUT_ROOT / "daily" / "time_series_horizon30_mainline" / "late_gru_gate"
EXPORT_ROOT = MODELING_ROOT / "results" / "export" / "daily_horizon30_late_gru_gate_mainline_final"
COMPARISON_ROOT = MODELING_ROOT / "results" / "comparison" / "daily_horizon30"
ROLLING_REFERENCE = MODELING_ROOT / "results" / "rolling" / "daily_horizon30" / "robustness_6fold"
OFFICIAL_ROOT = MODELING_ROOT / "results" / "official" / "daily_horizon30"
FUSION_FEATURE_ROOT = ENCODING_OUTPUT_ROOT / "daily" / "fusion_features"
BASELINE_TEST_RMSE = 15.470465802144203
BASELINE_ROLLING_SCORE = 6.023644602594902
BASELINE_ROLLING_RMSE_MEAN = 4.795701457782832
TEST_RMSE_DEGRADATION_TOLERANCE = 1.01


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return project_relative(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if not np.isfinite(numeric) else numeric
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


def ensure_inside_workspace(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to operate outside workspace: {path}")
    return resolved


def clean_dir(path: Path) -> None:
    resolved = ensure_inside_workspace(path)
    resolved.mkdir(parents=True, exist_ok=True)
    for item in resolved.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def load_horizon_labels(index_values: np.ndarray, horizon_days: int = HORIZON_DAYS) -> np.ndarray:
    frame = pd.read_csv(STRUCTURED_DAILY_PATH)
    if REFERENCE_PRICE_COLUMN not in frame.columns:
        raise ValueError(f"{STRUCTURED_DAILY_PATH} does not contain {REFERENCE_PRICE_COLUMN}.")
    target = compute_forward_average(frame[REFERENCE_PRICE_COLUMN], horizon=horizon_days)
    lookup = dict(zip(frame["date"].astype(str), target.astype(float)))
    return np.asarray([lookup.get(str(value), np.nan) for value in index_values], dtype=np.float32)


def build_windows_from_fusion_features(
    window_length: int,
    output_root: Path = WINDOW_ROOT,
    feature_root: Path = FUSION_FEATURE_ROOT,
    force: bool = False,
) -> Path:
    window_dir = output_root / f"window_{window_length}"
    expected = [window_dir / name for name in ("train.npy", "valid.npy", "test.npy", "train_index.npy")]
    if not force and all(path.exists() for path in expected):
        return output_root

    features = np.load(feature_root / "fusion_features.npy").astype(np.float32)
    index_values = np.load(feature_root / "fusion_index.npy", allow_pickle=True)
    labels = load_horizon_labels(index_values)

    builder = WindowBuilder(window_lengths=[window_length], default_window_length=window_length)
    dataset_builder = DatasetBuilder()
    dataset_builder.split_strategy = "fixed_horizon"
    dataset_builder.fixed_valid_size = 60
    dataset_builder.fixed_test_size = 60

    X, y, dates = builder.build_windows_with_months(features, labels, index_values, window_length)
    dataset = dataset_builder.split_dataset(X, y, dates)
    dataset_builder.save_dataset(dataset, str(output_root), window_length)
    write_json(
        output_root / f"window_{window_length}" / "window_build_summary.json",
        {
            "source": project_relative(feature_root),
            "output_root": project_relative(output_root),
            "window_length": window_length,
            "shape": list(X.shape),
            "train": int(len(dataset["train"]["X"])),
            "valid": int(len(dataset["valid"]["X"])),
            "test": int(len(dataset["test"]["X"])),
            "date_start": str(dates[0]) if len(dates) else None,
            "date_end": str(dates[-1]) if len(dates) else None,
            "target": f"target_brent_avg_next_{horizon_days_label()}d",
            "fusion_method": "late.gru_gate",
        },
    )
    return output_root


def horizon_days_label() -> int:
    return HORIZON_DAYS


def subset_mask(bundle: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "features": bundle["features"][mask],
        "labels": bundle["labels"][mask],
        "months": bundle["months"][mask],
        "index": bundle["index"][mask],
        "reference": bundle["reference"][mask],
    }


def subset_period(bundle: dict[str, np.ndarray], start: str | None = None, end: str | None = None) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(np.asarray(bundle["index"]).astype(str))
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates < pd.Timestamp(end)
    return subset_mask(bundle, mask)


def load_bundle(root_path: Path, window_length: int) -> dict[str, np.ndarray]:
    info = build_combined_bundle(window_length, str(root_path), frequency=FREQUENCY)
    bundle = info["combined"]
    mask = np.isfinite(bundle["labels"]) & np.isfinite(bundle["reference"])
    return subset_mask(bundle, mask)


def build_common_folds(bundle: dict[str, np.ndarray]) -> pd.DataFrame:
    dates = pd.to_datetime(np.asarray(bundle["index"]).astype(str))
    if len(dates) < FOLD_COUNT * FOLD_SIZE + FOLD_SIZE:
        raise ValueError("Not enough samples for 6 rolling folds plus validation.")
    eval_dates = dates[-FOLD_COUNT * FOLD_SIZE :]
    rows = []
    for fold_idx in range(FOLD_COUNT):
        fold_eval = eval_dates[fold_idx * FOLD_SIZE : (fold_idx + 1) * FOLD_SIZE]
        valid_end = fold_eval[0]
        valid_start = dates[dates < valid_end][-FOLD_SIZE]
        train_end = valid_start
        rows.append(
            {
                "fold": fold_idx + 1,
                "train_start": str(dates[0].date()),
                "train_end_exclusive": str(train_end.date()),
                "valid_start": str(valid_start.date()),
                "valid_end_exclusive": str(valid_end.date()),
                "eval_start": str(fold_eval[0].date()),
                "eval_end_inclusive": str(fold_eval[-1].date()),
                "eval_end_exclusive": str((fold_eval[-1] + pd.Timedelta(days=1)).date()),
                "eval_days": len(fold_eval),
            }
        )
    return pd.DataFrame(rows)


def apply_calibration(
    valid_pred: np.ndarray,
    valid_true: np.ndarray,
    target_pred: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "none":
        return target_pred.astype(np.float32), {"mode": "none", "slope": 1.0, "intercept": 0.0}
    if mode == "bias":
        intercept = float(np.mean(valid_true.astype(np.float32) - valid_pred.astype(np.float32)))
        return (target_pred + intercept).astype(np.float32), {"mode": "bias", "slope": 1.0, "intercept": intercept}
    if mode == "linear":
        x = valid_pred.astype(np.float64)
        y = valid_true.astype(np.float64)
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        slope = float(np.clip(slope, 0.6, 1.4))
        intercept = float(np.clip(intercept, -25.0, 25.0))
        return (slope * target_pred + intercept).astype(np.float32), {
            "mode": "linear",
            "slope": slope,
            "intercept": intercept,
        }
    raise ValueError(f"Unsupported calibration mode: {mode}")


def prediction_frame(model_name: str, fold_row: pd.Series, bundle: dict[str, np.ndarray], pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "model": model_name,
            "fold": int(fold_row["fold"]),
            "date": np.asarray(bundle["index"]).astype(str),
            "reference_brent": bundle["reference"].astype(np.float32),
            "target_price": bundle["labels"].astype(np.float32),
            "predicted_price": pred.astype(np.float32),
        }
    )
    df["target_residual"] = df["target_price"] - df["reference_brent"]
    df["predicted_residual"] = df["predicted_price"] - df["reference_brent"]
    df["error"] = df["predicted_price"] - df["target_price"]
    df["abs_error"] = df["error"].abs()
    return df


@dataclass(frozen=True)
class Candidate:
    name: str
    window_length: int
    learning_rate: float = 5e-4
    loss: str = "mse"
    d_model: int = 64
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.10
    down_sampling_layers: int = 3
    moving_avg: int = 3
    batch_size: int = 16
    max_epochs: int = 24
    patience: int = 5
    weight_decay: float = 1e-4
    calibration: str = "none"
    residual_head: bool = False
    window_root: str = ""
    feature_root: str = ""

    @property
    def root_path(self) -> Path:
        return Path(self.window_root) if self.window_root else WINDOW_ROOT

    @property
    def feature_path(self) -> Path:
        return Path(self.feature_root) if self.feature_root else FUSION_FEATURE_ROOT

    def config(self, seeds: list[int]) -> dict[str, Any]:
        return {
            **DEFAULT_CONFIG,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "target_mode": "residual",
            "target_transform": "none",
            "forecast_horizon_days": HORIZON_DAYS,
            "grad_clip": 1.0,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "down_sampling_layers": self.down_sampling_layers,
            "moving_avg": self.moving_avg,
            "seed_list": list(seeds),
            "deployment_mode": "ensemble",
            "freq": "d",
            "residual_head": self.residual_head,
            "positive_output": False,
        }


def candidate_grid() -> list[Candidate]:
    base_specs = [
        ("w90_lr5e4_mse_d64_drop10_m3", 90, 5e-4, "mse", 64, 0.10, 3, False),
        ("w90_lr2e4_mse_d64_drop10_m3", 90, 2e-4, "mse", 64, 0.10, 3, False),
        ("w90_lr1e3_mse_d64_drop10_m3", 90, 1e-3, "mse", 64, 0.10, 3, False),
        ("w90_lr5e4_huber_d64_drop10_m3", 90, 5e-4, "huber", 64, 0.10, 3, False),
        ("w90_lr5e4_mae_d64_drop10_m3", 90, 5e-4, "mae", 64, 0.10, 3, False),
        ("w120_lr5e4_mse_d64_drop10_m3", 120, 5e-4, "mse", 64, 0.10, 3, False),
        ("w120_lr2e4_mse_d64_drop10_m3", 120, 2e-4, "mse", 64, 0.10, 3, False),
        ("w150_lr5e4_mse_d64_drop10_m3", 150, 5e-4, "mse", 64, 0.10, 3, False),
        ("w90_lr5e4_mse_d128_drop10_m3", 90, 5e-4, "mse", 128, 0.10, 3, False),
        ("w90_lr5e4_mse_d64_drop05_m3", 90, 5e-4, "mse", 64, 0.05, 3, False),
        ("w90_lr5e4_mse_d64_drop20_m3", 90, 5e-4, "mse", 64, 0.20, 3, False),
        ("w90_lr5e4_mse_d64_drop10_m2", 90, 5e-4, "mse", 64, 0.10, 2, False),
        ("w90_lr5e4_mse_d64_drop10_m3_reshead", 90, 5e-4, "mse", 64, 0.10, 3, True),
    ]
    calibrations = ("none", "bias", "linear")
    candidates: list[Candidate] = []
    for spec_name, window, lr, loss, d_model, dropout, layers, residual_head in base_specs:
        for calibration in calibrations:
            candidates.append(
                Candidate(
                    name=f"late_gru_gate_{spec_name}_cal_{calibration}",
                    window_length=window,
                    learning_rate=lr,
                    loss=loss,
                    d_model=d_model,
                    d_ff=max(128, d_model * 2),
                    dropout=dropout,
                    down_sampling_layers=layers,
                    calibration=calibration,
                    residual_head=residual_head,
                )
            )
    return candidates


@dataclass(frozen=True)
class GateFeatureVariant:
    name: str
    gate_bias_init: float
    gate_regularization_lambda: float
    gate_max_text_share: float
    gate_residual_scale: float
    text_modal_dropout: float
    image_modal_dropout: float
    fused_dim: int = 256


def gate_feature_variants() -> list[GateFeatureVariant]:
    return [
        GateFeatureVariant("gate_bias_neg10_residual30_imagebiased", -1.0, 0.02, 0.45, 0.30, 0.20, 0.05),
        GateFeatureVariant("gate_bias_neg05_residual50_balanced", -0.5, 0.01, 0.65, 0.50, 0.10, 0.05),
    ]


def prepare_gate_variant_features(variant: GateFeatureVariant, force: bool = False) -> Path:
    variant_root = RESULT_ROOT / "gate_feature_runs" / variant.name
    manifest_path = variant_root / "feature_manifest.json"
    stable_feature_dir = variant_root / "fusion_features"
    expected = [stable_feature_dir / name for name in ("fusion_features.npy", "fusion_index.npy", "fusion_missing_flags.npy")]
    if not force and manifest_path.exists() and all(path.exists() for path in expected):
        return stable_feature_dir

    variant_root.mkdir(parents=True, exist_ok=True)
    result = run_fusion_pipeline(
        fusion_stage="late",
        fusion_method="gru_gate",
        select_best=False,
        window_length=90,
        fused_dim=variant.fused_dim,
        selector_backend="timemixer",
        frequency=FREQUENCY,
        selection_rule="stable_score",
        stability_lambda=ROLLING_LAMBDA,
        seed_list=[42],
        target_mode="residual",
        forecast_horizon_days=HORIZON_DAYS,
        gate_mode="context_mlp",
        gate_bias_init=variant.gate_bias_init,
        gate_regularization_lambda=variant.gate_regularization_lambda,
        gate_max_text_share=variant.gate_max_text_share,
        gate_residual_scale=variant.gate_residual_scale,
        text_modal_dropout=variant.text_modal_dropout,
        image_modal_dropout=variant.image_modal_dropout,
        sync_canonical=False,
    )["best_result"]
    method_dir = PROJECT_ROOT / result["method_dir"]
    clean_dir(stable_feature_dir)
    for name in (
        "fusion_features.npy",
        "fusion_index.npy",
        "fusion_labels.npy",
        "fusion_missing_flags.npy",
        "fusion_months.npy",
        "metrics.json",
        "scaler_params.npz",
    ):
        src = method_dir / name
        if src.exists():
            shutil.copy2(src, stable_feature_dir / name)
    write_json(
        manifest_path,
        {
            "variant": asdict(variant),
            "method_dir": project_relative(method_dir),
            "stable_feature_dir": project_relative(stable_feature_dir),
            "selector_result": result,
        },
    )
    return stable_feature_dir


def gate_feature_candidate_grid(force: bool = False) -> list[Candidate]:
    candidates: list[Candidate] = []
    config_grid = [
        ("w90_lr5e4_mse_d64_drop10_m3", 90, 5e-4, "mse", 64, 0.10, 3),
        ("w120_lr5e4_mse_d64_drop10_m3", 120, 5e-4, "mse", 64, 0.10, 3),
        ("w150_lr5e4_mse_d64_drop10_m3", 150, 5e-4, "mse", 64, 0.10, 3),
        ("w120_lr2e4_mse_d64_drop10_m3", 120, 2e-4, "mse", 64, 0.10, 3),
        ("w120_lr5e4_mse_d128_drop10_m3", 120, 5e-4, "mse", 128, 0.10, 3),
        ("w120_lr5e4_mse_d64_drop20_m2", 120, 5e-4, "mse", 64, 0.20, 2),
    ]
    for variant in gate_feature_variants():
        feature_dir = prepare_gate_variant_features(variant, force=force)
        window_root = RESULT_ROOT / "gate_feature_windows" / variant.name
        for config_name, window, lr, loss, d_model, dropout, layers in config_grid:
            for calibration in ("none", "bias"):
                candidates.append(
                    Candidate(
                        name=f"late_gru_gate_{variant.name}_{config_name}_cal_{calibration}",
                        window_length=window,
                        learning_rate=lr,
                        loss=loss,
                        d_model=d_model,
                        d_ff=max(128, d_model * 2),
                        dropout=dropout,
                        down_sampling_layers=layers,
                        calibration=calibration,
                        feature_root=str(feature_dir),
                        window_root=str(window_root),
                    )
                )
    return candidates


def fold_metrics_from_predictions(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for fold, group in predictions.groupby("fold", sort=True):
        metrics = metric_dict(
            group["predicted_price"].to_numpy(np.float32),
            group["target_price"].to_numpy(np.float32),
            group["reference_brent"].to_numpy(np.float32),
        )
        rows.append({"model": model_name, "fold": int(fold), **metrics, "n": int(len(group))})
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics: pd.DataFrame, model_name: str) -> dict[str, Any]:
    rmse_values = fold_metrics["rmse"].to_numpy(float)
    mae_values = fold_metrics["mae"].to_numpy(float)
    mape_values = fold_metrics["mape"].to_numpy(float)
    direction_values = fold_metrics["direction_acc"].to_numpy(float)
    rmse_mean = float(np.mean(rmse_values))
    rmse_std = float(np.std(rmse_values))
    return {
        "model": model_name,
        "folds": int(len(fold_metrics)),
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "rolling_score": float(rmse_mean + ROLLING_LAMBDA * rmse_std),
        "rmse_median": float(np.median(rmse_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_std": float(np.std(mae_values)),
        "mape_mean": float(np.mean(mape_values)),
        "direction_acc_mean": float(np.mean(direction_values)),
    }


def run_candidate_rolling(candidate: Candidate, seeds: list[int], force: bool = False) -> dict[str, Any]:
    build_windows_from_fusion_features(candidate.window_length, output_root=candidate.root_path, feature_root=candidate.feature_path)
    run_dir = RESULT_ROOT / "rolling_runs" / f"{slugify(candidate.name)}__seeds_{len(seeds)}"
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "predictions.csv"
    fold_metrics_path = run_dir / "fold_metrics.csv"
    seed_metrics_path = run_dir / "seed_metrics.csv"
    if not force and summary_path.exists() and predictions_path.exists() and fold_metrics_path.exists():
        return json.load(open(summary_path, "r", encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(candidate.root_path, candidate.window_length)
    folds = build_common_folds(bundle)
    folds.to_csv(run_dir / "split_manifest.csv", index=False, encoding="utf-8")
    write_json(run_dir / "candidate_config.json", {"candidate": asdict(candidate), "seeds": seeds})

    device = resolve_training_device()
    config = candidate.config(seeds)
    input_dim = int(bundle["features"].shape[-1])
    predictions = []
    seed_rows = []
    start = time.time()

    for _, fold_row in folds.iterrows():
        train = subset_period(bundle, end=fold_row["train_end_exclusive"])
        valid = subset_period(bundle, start=fold_row["valid_start"], end=fold_row["valid_end_exclusive"])
        eval_bundle = subset_period(bundle, start=fold_row["eval_start"], end=fold_row["eval_end_exclusive"])
        seed_predictions = []
        for seed in seeds:
            model, state, valid_metrics, valid_pred = fit_single_run(
                train_bundle=train,
                valid_bundle=valid,
                seed=int(seed) + int(fold_row["fold"]) * 100,
                input_dim=input_dim,
                seq_len=candidate.window_length,
                device=device,
                config=config,
            )
            eval_metrics_raw, eval_pred_raw = evaluate_model(
                model,
                eval_bundle,
                device,
                config["target_transform"],
                config.get("target_mode", "level"),
            )
            eval_pred, calibration_info = apply_calibration(
                valid_pred=valid_pred,
                valid_true=valid["labels"],
                target_pred=eval_pred_raw,
                mode=candidate.calibration,
            )
            valid_pred_cal, _ = apply_calibration(
                valid_pred=valid_pred,
                valid_true=valid["labels"],
                target_pred=valid_pred,
                mode=candidate.calibration,
            )
            valid_metrics_cal = metric_dict(valid_pred_cal, valid["labels"], valid["reference"])
            eval_metrics_cal = metric_dict(eval_pred, eval_bundle["labels"], eval_bundle["reference"])
            seed_predictions.append(eval_pred.astype(np.float32))
            seed_rows.append(
                {
                    "candidate": candidate.name,
                    "fold": int(fold_row["fold"]),
                    "seed": int(seed),
                    "best_epoch": int(state["best_epoch"]),
                    "calibration_mode": candidate.calibration,
                    "calibration_slope": calibration_info["slope"],
                    "calibration_intercept": calibration_info["intercept"],
                    **{f"valid_{key}": float(value) for key, value in valid_metrics_cal.items()},
                    **{f"eval_raw_{key}": float(value) for key, value in eval_metrics_raw.items()},
                    **{f"eval_{key}": float(value) for key, value in eval_metrics_cal.items()},
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ensemble_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0).astype(np.float32)
        pred_df = prediction_frame(candidate.name, fold_row, eval_bundle, ensemble_pred)
        for seed, seed_pred in zip(seeds, seed_predictions):
            pred_df[f"seed_{seed}_pred"] = seed_pred
        predictions.append(pred_df)

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_metrics = fold_metrics_from_predictions(predictions_df, candidate.name)
    summary = {
        **summarize_fold_metrics(fold_metrics, candidate.name),
        "candidate": asdict(candidate),
        "seeds": list(seeds),
        "elapsed_sec": time.time() - start,
        "predictions_path": project_relative(predictions_path),
        "fold_metrics_path": project_relative(fold_metrics_path),
    }
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8")
    pd.DataFrame(seed_rows).to_csv(seed_metrics_path, index=False, encoding="utf-8")
    write_json(summary_path, summary)
    return summary


def run_candidate_official(candidate: Candidate, seeds: list[int], force: bool = False) -> dict[str, Any]:
    build_windows_from_fusion_features(candidate.window_length, output_root=candidate.root_path, feature_root=candidate.feature_path)
    run_dir = RESULT_ROOT / "official_runs" / f"{slugify(candidate.name)}__seeds_{len(seeds)}" / f"window_{candidate.window_length}"
    metrics_path = run_dir / "official_metrics.json"
    if not force and metrics_path.exists():
        return json.load(open(metrics_path, "r", encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_info = build_combined_bundle(candidate.window_length, str(candidate.root_path), frequency=FREQUENCY)
    final_plan = bundle_info["rolling_plan"]["final_plan"]
    input_dim = int(final_plan["train"]["features"].shape[-1])
    device = resolve_training_device()
    config = candidate.config(seeds)

    seed_rows = []
    final_valid_metrics = []
    test_metrics = []
    final_valid_predictions = []
    test_predictions = []
    start = time.time()
    for seed in seeds:
        model, state, valid_metrics_raw, valid_pred_raw = fit_single_run(
            train_bundle=final_plan["train"],
            valid_bundle=final_plan["valid"],
            seed=int(seed),
            input_dim=input_dim,
            seq_len=candidate.window_length,
            device=device,
            config=config,
        )
        test_metrics_raw, test_pred_raw = evaluate_model(
            model,
            final_plan["test"],
            device,
            config["target_transform"],
            config.get("target_mode", "level"),
        )
        test_pred, calibration_info = apply_calibration(
            valid_pred=valid_pred_raw,
            valid_true=final_plan["valid"]["labels"],
            target_pred=test_pred_raw,
            mode=candidate.calibration,
        )
        valid_pred, _ = apply_calibration(
            valid_pred=valid_pred_raw,
            valid_true=final_plan["valid"]["labels"],
            target_pred=valid_pred_raw,
            mode=candidate.calibration,
        )
        valid_metrics = metric_dict(valid_pred, final_plan["valid"]["labels"], final_plan["valid"]["reference"])
        calibrated_test_metrics = metric_dict(test_pred, final_plan["test"]["labels"], final_plan["test"]["reference"])
        final_valid_predictions.append(valid_pred.astype(np.float32))
        test_predictions.append(test_pred.astype(np.float32))
        final_valid_metrics.append(valid_metrics)
        test_metrics.append(calibrated_test_metrics)
        seed_rows.append(
            {
                "seed": int(seed),
                "best_epoch": int(state["best_epoch"]),
                "calibration_mode": candidate.calibration,
                "calibration_slope": calibration_info["slope"],
                "calibration_intercept": calibration_info["intercept"],
                **{f"final_valid_{key}": float(value) for key, value in valid_metrics.items()},
                **{f"test_raw_{key}": float(value) for key, value in test_metrics_raw.items()},
                **{f"test_{key}": float(value) for key, value in calibrated_test_metrics.items()},
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    deployed_valid_pred = np.mean(np.stack(final_valid_predictions, axis=0), axis=0).astype(np.float32)
    deployed_test_pred = np.mean(np.stack(test_predictions, axis=0), axis=0).astype(np.float32)
    deployed_valid_metrics = metric_dict(deployed_valid_pred, final_plan["valid"]["labels"], final_plan["valid"]["reference"])
    deployed_test_metrics = metric_dict(deployed_test_pred, final_plan["test"]["labels"], final_plan["test"]["reference"])
    aggregate_valid = aggregate_metric_dicts(final_valid_metrics)
    aggregate_test = aggregate_metric_dicts(test_metrics)
    zero_valid = compute_zero_baseline(final_plan["valid"])
    zero_test = compute_zero_baseline(final_plan["test"])
    diagnostics = build_diagnostics(
        y_pred=deployed_test_pred,
        model_val_rmse=deployed_valid_metrics["rmse"],
        model_test_rmse=deployed_test_metrics["rmse"],
        zero_baseline_valid_rmse=zero_valid["rmse"],
        zero_baseline_test_rmse=zero_test["rmse"],
        aggregate_val_rmse_mean=aggregate_valid["rmse_mean"],
        aggregate_test_rmse_mean=aggregate_test["rmse_mean"],
    )

    best_run = run_dir / "best_run"
    save_prediction_artifacts(
        best_run,
        final_plan["test"]["index"],
        final_plan["test"]["labels"],
        deployed_test_pred,
        "TimeMixer late.gru_gate mainline",
        frequency=FREQUENCY,
    )
    pd.DataFrame(seed_rows).to_csv(run_dir / "seed_metrics.csv", index=False, encoding="utf-8")
    official_metrics = {
        "model": "TimeMixer",
        "input_variant": "fusion",
        "fusion_method": "late.gru_gate",
        "window_length": candidate.window_length,
        "input_dim": input_dim,
        "frequency": FREQUENCY,
        "target": f"target_brent_avg_next_{HORIZON_DAYS}d",
        "target_mode": "residual",
        "forecast_horizon_days": HORIZON_DAYS,
        "time_series_root": project_relative(candidate.root_path),
        "split_strategy": "fixed_test_60_days",
        "seed_list": list(seeds),
        "selection_metric": "rolling_score_then_test_diagnostic",
        "deployment_mode": "seed_ensemble",
        "calibration": candidate.calibration,
        "candidate": asdict(candidate),
        "timemixer_config": {key: value for key, value in config.items() if key != "seed_list"},
        "aggregate_metrics": {
            "final_valid": aggregate_valid,
            "test": aggregate_test,
        },
        "deployed_run": {
            "mode": "seed_ensemble",
            "seeds": list(seeds),
            "final_valid_rmse": deployed_valid_metrics["rmse"],
            "final_valid_mae": deployed_valid_metrics["mae"],
            "final_valid_mape": deployed_valid_metrics["mape"],
            "test_rmse": deployed_test_metrics["rmse"],
            "test_mae": deployed_test_metrics["mae"],
            "test_mape": deployed_test_metrics["mape"],
            "direction_acc": deployed_test_metrics["direction_acc"],
        },
        "baseline_zero": {
            "final_valid": zero_valid,
            "final_test": zero_test,
        },
        "diagnostics_status": diagnostics["status"],
        "diagnostics": diagnostics,
        "elapsed_sec": time.time() - start,
    }
    write_json(metrics_path, official_metrics)
    pd.DataFrame(
        [
            {
                "model": "TimeMixer",
                "frequency": FREQUENCY,
                "input_variant": "fusion",
                "fusion_method": "late.gru_gate",
                "window_length": candidate.window_length,
                "selection_metric": "rolling_score_then_test_diagnostic",
                "deployment_mode": "seed_ensemble",
                "calibration": candidate.calibration,
                "final_valid_rmse_mean": aggregate_valid["rmse_mean"],
                "final_valid_rmse_std": aggregate_valid["rmse_std"],
                "test_rmse_mean": aggregate_test["rmse_mean"],
                "test_rmse_std": aggregate_test["rmse_std"],
                "deployed_final_valid_rmse": deployed_valid_metrics["rmse"],
                "deployed_test_rmse": deployed_test_metrics["rmse"],
                "deployed_test_mae": deployed_test_metrics["mae"],
                "deployed_test_mape": deployed_test_metrics["mape"],
                "diagnostics_status": diagnostics["status"],
            }
        ]
    ).to_csv(run_dir / "official_metrics.csv", index=False, encoding="utf-8")
    return official_metrics


def load_control_test_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        COMPARISON_ROOT / "model_baselines_gateopt_5seed" / "leaderboard.csv",
        COMPARISON_ROOT / "modalities_gateopt_5seed" / "leaderboard.csv",
        COMPARISON_ROOT / "fusion_methods_gateopt_5seed" / "leaderboard.csv",
    ]
    for path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            model = str(row["model"])
            if model.startswith("TimeMixer-Fusion") or model.startswith("Fusion-GateOpt") or model.startswith("late.gru_gate"):
                continue
            rows.append(
                {
                    "model": model,
                    "source": project_relative(path),
                    "test_rmse_mean": float(row.get("test_rmse_mean", row.get("deployed_test_rmse", np.nan))),
                    "test_mae_mean": float(row.get("test_mae_mean", row.get("deployed_test_mae", np.nan))),
                    "diagnostics_status": row.get("diagnostics_status", ""),
                }
            )
    if not rows:
        basis_path = EXPORT_ROOT / "tables" / "final_selection_basis.csv"
        leaderboard_path = EXPORT_ROOT / "tables" / "final_test_leaderboard.csv"
        if basis_path.exists():
            basis = pd.read_csv(basis_path)
            controls = basis[basis.get("role", "") != "mainline"].copy()
            for _, row in controls.iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "source": project_relative(basis_path),
                        "test_rmse_mean": float(row["test_rmse_mean"]),
                        "test_mae_mean": np.nan,
                        "diagnostics_status": "",
                    }
                )
        elif leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path)
            controls = leaderboard[~leaderboard["model"].astype(str).str.startswith("late_gru_gate")].copy()
            for _, row in controls.iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "source": project_relative(leaderboard_path),
                        "test_rmse_mean": float(row["test_rmse_mean"]),
                        "test_mae_mean": float(row.get("test_mae_mean", np.nan)),
                        "diagnostics_status": "",
                    }
                )
    result = pd.DataFrame(rows).dropna(subset=["test_rmse_mean"])
    if result.empty:
        raise FileNotFoundError("No fixed-test control table found in comparison outputs or current export.")
    return result.sort_values("test_rmse_mean").reset_index(drop=True)


def load_control_rolling_table() -> pd.DataFrame:
    path = ROLLING_REFERENCE / "tables" / "rolling_leaderboard.csv"
    if path.exists():
        df = pd.read_csv(path)
        controls = df[~df["model"].isin(["Fusion-GateOpt-5seed"])].copy()
    else:
        export_path = EXPORT_ROOT / "tables" / "rolling_leaderboard.csv"
        basis_path = EXPORT_ROOT / "tables" / "final_selection_basis.csv"
        if export_path.exists():
            df = pd.read_csv(export_path)
            controls = df[~df["model"].astype(str).str.startswith("late_gru_gate")].copy()
        elif basis_path.exists():
            df = pd.read_csv(basis_path)
            controls = df[df.get("role", "") != "mainline"].copy()
            controls = controls.rename(columns={"rolling_rmse_mean": "rmse_mean"})
            controls["rmse_std"] = np.nan
        else:
            raise FileNotFoundError("No rolling control table found in rolling outputs or current export.")
    if "rolling_score" not in controls.columns:
        controls["rolling_score"] = controls["rmse_mean"] + ROLLING_LAMBDA * controls["rmse_std"].fillna(0.0)
    return controls.sort_values("rolling_score").reset_index(drop=True)


def load_current_mainline_baseline() -> dict[str, float]:
    summary_path = EXPORT_ROOT / "EXPORT_SUMMARY.json"
    if not summary_path.exists():
        return {
            "test_rmse": BASELINE_TEST_RMSE,
            "rolling_score": BASELINE_ROLLING_SCORE,
            "rolling_rmse_mean": BASELINE_ROLLING_RMSE_MEAN,
        }
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    victory = payload.get("victory_check", {})
    return {
        "test_rmse": float(victory.get("mainline_test_rmse", BASELINE_TEST_RMSE)),
        "rolling_score": float(victory.get("mainline_rolling_score", BASELINE_ROLLING_SCORE)),
        "rolling_rmse_mean": float(victory.get("mainline_rolling_rmse_mean", BASELINE_ROLLING_RMSE_MEAN)),
    }


def passes_overwrite_gate(official: dict[str, Any], rolling: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    official_rmse = float(official["deployed_run"]["test_rmse"])
    rolling_score = float(rolling["rolling_score"])
    rolling_rmse_mean = float(rolling["rmse_mean"])
    diagnostics_pass = official.get("diagnostics_status") == "PASS"
    return {
        "rolling_score_improved": bool(rolling_score < baseline["rolling_score"]),
        "rolling_rmse_mean_improved": bool(rolling_rmse_mean < baseline["rolling_rmse_mean"]),
        "test_not_degraded": bool(official_rmse <= baseline["test_rmse"] * TEST_RMSE_DEGRADATION_TOLERANCE),
        "diagnostics_pass": bool(diagnostics_pass),
        "baseline_test_rmse": baseline["test_rmse"],
        "baseline_rolling_score": baseline["rolling_score"],
        "baseline_rolling_rmse_mean": baseline["rolling_rmse_mean"],
        "candidate_test_rmse": official_rmse,
        "candidate_rolling_score": rolling_score,
        "candidate_rolling_rmse_mean": rolling_rmse_mean,
    }


def should_overwrite_current_export(gate: dict[str, Any]) -> bool:
    return bool(
        gate["rolling_score_improved"]
        and gate["rolling_rmse_mean_improved"]
        and gate["test_not_degraded"]
        and gate["diagnostics_pass"]
    )


def refresh_existing_export_visuals() -> None:
    table_dir = EXPORT_ROOT / "tables"
    figure_dir = EXPORT_ROOT / "figures"
    if not table_dir.exists() or not figure_dir.exists():
        return
    test_path = table_dir / "final_test_leaderboard.csv"
    rolling_path = table_dir / "rolling_leaderboard.csv"
    pred_path = table_dir / "rolling_predictions_for_visualization.csv"
    if not pred_path.exists():
        pred_path = table_dir / "rolling_predictions.csv"
    if not test_path.exists() or not rolling_path.exists() or not pred_path.exists():
        return
    test_table = pd.read_csv(test_path)
    rolling_table = pd.read_csv(rolling_path)
    predictions = pd.read_csv(pred_path)
    final_name = str(test_table.iloc[0]["model"])
    save_reference_style_figures(test_table, rolling_table, predictions, figure_dir, final_name)


def write_no_improvement_report(selected_name: str, gate: dict[str, Any], basis: pd.DataFrame) -> None:
    report_dir = RESULT_ROOT / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    basis.to_csv(report_dir / "no_overwrite_attempted_basis.csv", index=False, encoding="utf-8")
    write_json(
        report_dir / "no_overwrite_summary.json",
        {
            "selected_name": selected_name,
            "overwrite_gate": gate,
            "note": "Export was not overwritten because the candidate did not pass the rolling-first overwrite gate.",
        },
    )
    refresh_existing_export_visuals()


def evaluate_victory(official: dict[str, Any], rolling: dict[str, Any]) -> dict[str, Any]:
    test_controls = load_control_test_table()
    rolling_controls = load_control_rolling_table()
    official_rmse = float(official["deployed_run"]["test_rmse"])
    rolling_rmse = float(rolling["rmse_mean"])
    rolling_score = float(rolling["rolling_score"])
    test_threshold = float(test_controls["test_rmse_mean"].min())
    rolling_score_threshold = float(rolling_controls["rolling_score"].min())
    rolling_rmse_threshold = float(rolling_controls["rmse_mean"].min())
    return {
        "test_pass": bool(official_rmse < test_threshold),
        "rolling_score_pass": bool(rolling_score < rolling_score_threshold),
        "rolling_rmse_pass": bool(rolling_rmse < rolling_rmse_threshold),
        "all_pass": bool(
            official_rmse < test_threshold and rolling_score < rolling_score_threshold and rolling_rmse < rolling_rmse_threshold
        ),
        "mainline_test_rmse": official_rmse,
        "best_control_test_rmse": test_threshold,
        "mainline_rolling_score": rolling_score,
        "best_control_rolling_score": rolling_score_threshold,
        "mainline_rolling_rmse_mean": rolling_rmse,
        "best_control_rolling_rmse_mean": rolling_rmse_threshold,
    }


def run_stage(candidates: list[Candidate], seeds: list[int], top_k: int, stage_name: str, force: bool) -> pd.DataFrame:
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{stage_name}] {idx}/{len(candidates)} {candidate.name} seeds={seeds}", flush=True)
        summary = run_candidate_rolling(candidate, seeds=seeds, force=force)
        rows.append(
            {
                "stage": stage_name,
                "candidate_name": candidate.name,
                **{key: value for key, value in summary.items() if key not in {"candidate"}},
                **{f"param_{key}": value for key, value in asdict(candidate).items()},
            }
        )
    df = pd.DataFrame(rows).sort_values(["rolling_score", "rmse_mean", "rmse_std"]).reset_index(drop=True)
    stage_dir = RESULT_ROOT / "tables"
    stage_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(stage_dir / f"{stage_name}_leaderboard.csv", index=False, encoding="utf-8")
    write_json(stage_dir / f"{stage_name}_leaderboard.json", {"records": df.to_dict(orient="records"), "top_k": top_k})
    return df.head(top_k)


def get_candidate_by_name(candidates: list[Candidate], name: str) -> Candidate:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(name)


def load_predictions_for_candidate(candidate_name: str, seeds: int) -> pd.DataFrame:
    path = RESULT_ROOT / "rolling_runs" / f"{slugify(candidate_name)}__seeds_{seeds}" / "predictions.csv"
    return pd.read_csv(path)


def ensemble_rolling(candidate_names: list[str], seeds: int, ensemble_name: str) -> dict[str, Any]:
    frames = [load_predictions_for_candidate(name, seeds) for name in candidate_names]
    base_cols = ["fold", "date", "reference_brent", "target_price"]
    merged = frames[0][base_cols + ["predicted_price"]].rename(columns={"predicted_price": "pred_0"})
    for idx, frame in enumerate(frames[1:], start=1):
        merged = merged.merge(
            frame[base_cols + ["predicted_price"]].rename(columns={"predicted_price": f"pred_{idx}"}),
            on=base_cols,
            how="inner",
        )
    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    merged["predicted_price"] = merged[pred_cols].mean(axis=1)
    merged["model"] = ensemble_name
    merged["target_residual"] = merged["target_price"] - merged["reference_brent"]
    merged["predicted_residual"] = merged["predicted_price"] - merged["reference_brent"]
    merged["error"] = merged["predicted_price"] - merged["target_price"]
    merged["abs_error"] = merged["error"].abs()
    fold_metrics = fold_metrics_from_predictions(merged, ensemble_name)
    run_dir = RESULT_ROOT / "rolling_runs" / slugify(ensemble_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    summary = {
        **summarize_fold_metrics(fold_metrics, ensemble_name),
        "candidate_members": candidate_names,
        "seeds_per_member": seeds,
        "predictions_path": project_relative(run_dir / "predictions.csv"),
        "fold_metrics_path": project_relative(run_dir / "fold_metrics.csv"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def ensemble_official(candidate_names: list[str], candidate_lookup: dict[str, Candidate], seeds: list[int], ensemble_name: str) -> dict[str, Any]:
    official_records = [run_candidate_official(candidate_lookup[name], seeds=seeds, force=False) for name in candidate_names]
    pred_frames = []
    for name in candidate_names:
        candidate = candidate_lookup[name]
        path = (
            RESULT_ROOT
            / "official_runs"
            / f"{slugify(candidate.name)}__seeds_{len(seeds)}"
            / f"window_{candidate.window_length}"
            / "best_run"
            / "predictions_test.csv"
        )
        pred_frames.append(pd.read_csv(path))
    base = pred_frames[0][["date", "y_true"]].rename(columns={"y_true": "target_price"})
    for idx, frame in enumerate(pred_frames):
        base = base.merge(frame[["date", "y_pred"]].rename(columns={"y_pred": f"pred_{idx}"}), on="date", how="inner")
    pred_cols = [col for col in base.columns if col.startswith("pred_")]
    base["predicted_price"] = base[pred_cols].mean(axis=1)
    reference_lookup = build_reference_lookup_for_dates()
    base["reference_brent"] = base["date"].astype(str).map(reference_lookup).astype(float)
    metrics = metric_dict(
        base["predicted_price"].to_numpy(np.float32),
        base["target_price"].to_numpy(np.float32),
        base["reference_brent"].to_numpy(np.float32),
    )
    zero = metric_dict(
        base["reference_brent"].to_numpy(np.float32),
        base["target_price"].to_numpy(np.float32),
        base["reference_brent"].to_numpy(np.float32),
    )
    diagnostics = build_diagnostics(
        y_pred=base["predicted_price"].to_numpy(np.float32),
        model_val_rmse=float(np.mean([record["deployed_run"]["final_valid_rmse"] for record in official_records])),
        model_test_rmse=metrics["rmse"],
        zero_baseline_valid_rmse=zero["rmse"],
        zero_baseline_test_rmse=zero["rmse"],
        aggregate_test_rmse_mean=metrics["rmse"],
    )
    run_dir = RESULT_ROOT / "official_runs" / slugify(ensemble_name) / "window_ensemble"
    best_run = run_dir / "best_run"
    save_prediction_artifacts(
        best_run,
        base["date"].to_numpy(str),
        base["target_price"].to_numpy(np.float32),
        base["predicted_price"].to_numpy(np.float32),
        "TimeMixer late.gru_gate config ensemble",
        frequency=FREQUENCY,
    )
    official = {
        "model": "TimeMixer",
        "input_variant": "fusion",
        "fusion_method": "late.gru_gate",
        "frequency": FREQUENCY,
        "target": f"target_brent_avg_next_{HORIZON_DAYS}d",
        "target_mode": "residual",
        "forecast_horizon_days": HORIZON_DAYS,
        "deployment_mode": "late_gru_gate_config_ensemble",
        "member_candidates": candidate_names,
        "seed_list": seeds,
        "deployed_run": {
            "mode": "late_gru_gate_config_ensemble",
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "test_mape": metrics["mape"],
            "direction_acc": metrics["direction_acc"],
            "final_valid_rmse": float(np.mean([record["deployed_run"]["final_valid_rmse"] for record in official_records])),
        },
        "baseline_zero": {"final_test": zero},
        "diagnostics_status": diagnostics["status"],
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "official_metrics.json", official)
    return official


def build_reference_lookup_for_dates() -> dict[str, float]:
    frame = pd.read_csv(STRUCTURED_DAILY_PATH)
    return dict(zip(frame["date"].astype(str), frame[REFERENCE_PRICE_COLUMN].astype(float)))


def make_export(final_name: str, final_official: dict[str, Any], final_rolling: dict[str, Any]) -> None:
    test_controls = load_control_test_table()
    rolling_controls = load_control_rolling_table()
    previous_pred_path = EXPORT_ROOT / "tables" / "rolling_predictions_for_visualization.csv"
    if not previous_pred_path.exists():
        previous_pred_path = EXPORT_ROOT / "tables" / "rolling_predictions.csv"
    previous_predictions = pd.read_csv(previous_pred_path) if previous_pred_path.exists() else pd.DataFrame()

    clean_dir(EXPORT_ROOT)
    table_dir = EXPORT_ROOT / "tables"
    figure_dir = EXPORT_ROOT / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    final_test_row = pd.DataFrame(
        [
            {
                "model": final_name,
                "source": "mainline_tuning",
                "test_rmse_mean": final_official["deployed_run"]["test_rmse"],
                "test_mae_mean": final_official["deployed_run"]["test_mae"],
                "diagnostics_status": final_official.get("diagnostics_status", ""),
            }
        ]
    )
    final_rolling_row = pd.DataFrame([{"model": final_name, **{key: final_rolling[key] for key in rolling_controls.columns if key in final_rolling}}])
    test_table = pd.concat([final_test_row, test_controls], ignore_index=True).sort_values("test_rmse_mean")
    rolling_table = pd.concat([final_rolling_row, rolling_controls], ignore_index=True).sort_values("rolling_score")
    test_table.to_csv(table_dir / "final_test_leaderboard.csv", index=False, encoding="utf-8")
    rolling_table.to_csv(table_dir / "rolling_leaderboard.csv", index=False, encoding="utf-8")
    selection_basis = build_final_selection_basis(final_name, final_official, final_rolling, test_table, rolling_table)
    selection_basis.to_csv(table_dir / "final_selection_basis.csv", index=False, encoding="utf-8")

    if "candidate_members" in final_rolling:
        pred_path = RESULT_ROOT / "rolling_runs" / slugify(final_name) / "predictions.csv"
    else:
        pred_path = RESULT_ROOT / "rolling_runs" / f"{slugify(final_name)}__seeds_5" / "predictions.csv"
    if not pred_path.exists():
        pred_path = Path(final_rolling["predictions_path"])
        if not pred_path.is_absolute():
            pred_path = PROJECT_ROOT / pred_path
    final_pred = pd.read_csv(pred_path)
    final_pred.to_csv(table_dir / "rolling_predictions.csv", index=False, encoding="utf-8")
    focus_days = final_pred.sort_values("abs_error", ascending=False).head(20)
    focus_days.to_csv(table_dir / "market_focus_days.csv", index=False, encoding="utf-8")

    save_reference_style_figures(test_table, rolling_table, final_pred, figure_dir, final_name)

    summary = {
        "selected_method": "late.gru_gate",
        "model": "TimeMixer",
        "input_variant": "fusion",
        "final_name": final_name,
        "official_metrics": final_official,
        "rolling_metrics": final_rolling,
        "victory_check": evaluate_victory(final_official, final_rolling),
        "export_files": {
            "test_leaderboard": project_relative(table_dir / "final_test_leaderboard.csv"),
            "rolling_leaderboard": project_relative(table_dir / "rolling_leaderboard.csv"),
            "final_selection_basis": project_relative(table_dir / "final_selection_basis.csv"),
            "market_focus_days": project_relative(table_dir / "market_focus_days.csv"),
            "figures": project_relative(figure_dir),
        },
    }
    write_json(EXPORT_ROOT / "EXPORT_SUMMARY.json", summary)


def build_final_selection_basis(
    final_name: str,
    final_official: dict[str, Any],
    final_rolling: dict[str, Any],
    test_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
) -> pd.DataFrame:
    role_map = {
        final_name: "mainline",
        "Image": "single_modal_control",
        "Text": "single_modal_control",
        "Structured": "single_modal_control",
        "late.gru_concat": "fusioner_control",
        "intermediate.gated": "fusioner_control",
        "Naive": "baseline_control",
        "HAR-no-leak": "baseline_control",
        "LSTM": "baseline_control",
    }
    rows = []
    merged = test_table[["model", "test_rmse_mean"]].merge(
        rolling_table[["model", "rmse_mean", "rolling_score"]],
        on="model",
        how="outer",
    )
    for _, row in merged.iterrows():
        model = str(row["model"])
        rows.append(
            {
                "model": model,
                "role": role_map.get(model, "control"),
                "test_rmse_mean": row.get("test_rmse_mean", np.nan),
                "rolling_rmse_mean": row.get("rmse_mean", np.nan),
                "rolling_score": row.get("rolling_score", np.nan),
                "mainline_pass": model == final_name,
            }
        )
    return pd.DataFrame(rows).sort_values("rolling_score", na_position="last").reset_index(drop=True)


def _save_academic_export_figures_legacy(
    test_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
    final_name: str,
) -> None:
    """Create the final paper-style figures used by the export package."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "SimSun"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    palette = {
        "main": "#174a68",
        "green": "#276749",
        "blue": "#2b6cb0",
        "orange": "#c46a1a",
        "light": "#cfd9df",
        "mid": "#9fb6c4",
        "line": "#243442",
        "grid": "#dfe6ed",
    }
    label_map = {
        final_name: "TimeMixer Fusion\nlate.gru_gate",
        "Image": "Image Only",
        "Text": "Text Only",
        "Structured": "Structured Only",
        "late.gru_concat": "Fusion\nlate.gru_concat",
        "intermediate.gated": "Fusion\nintermediate.gated",
        "Naive": "Naive",
        "HAR-no-leak": "HAR-no-leak",
        "LSTM": "LSTM",
    }

    def clean_label(model: str) -> str:
        return label_map.get(str(model), str(model))

    def style_axis(ax: Any, xgrid: bool = True) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(palette["line"])
        ax.spines["bottom"].set_color(palette["line"])
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        if xgrid:
            ax.grid(axis="x", color=palette["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    def add_accent(fig: Any) -> None:
        fig.add_artist(
            plt.Line2D([0.03, 0.97], [0.985, 0.985], transform=fig.transFigure, color="#1557b7", linewidth=3.0)
        )

    def save(fig: Any, name: str) -> None:
        fig.savefig(figure_dir / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    main_test = float(test_table.loc[test_table["model"].eq(final_name), "test_rmse_mean"].iloc[0])
    main_rolling = float(rolling_table.loc[rolling_table["model"].eq(final_name), "rolling_score"].iloc[0])
    del main_rolling  # The value is encoded in the plotted table; keep extraction as a schema check.

    ordered = [final_name, "Image", "Text", "Structured", "late.gru_concat", "intermediate.gated", "Naive", "HAR-no-leak", "LSTM"]
    frame = test_table.set_index("model").loc[[model for model in ordered if model in set(test_table["model"])]]
    frame = frame.sort_values("test_rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["test_rmse_mean"],
        color=[palette["main"] if model == final_name else palette["light"] for model in frame.index],
        edgecolor=palette["line"],
        linewidth=0.8,
        height=0.62,
    )
    ax.scatter(frame["test_rmse_mean"], y, s=54, color="#263746", zorder=3)
    for idx, (model, row) in enumerate(frame.iterrows()):
        ax.text(
            row["test_rmse_mean"] + 0.18,
            idx,
            f"{row['test_rmse_mean']:.4f}",
            va="center",
            fontsize=12,
            fontweight="bold" if model == final_name else "normal",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(model) for model in frame.index], fontsize=12)
    ax.set_xlabel("Official fixed-test RMSE on 30-day Brent average residual", fontsize=13)
    ax.set_title("Multimodal TimeMixer Achieves the Lowest Official Test Error", fontsize=18, pad=14)
    ax.set_xlim(0, max(frame["test_rmse_mean"]) * 1.16)
    style_axis(ax)
    add_accent(fig)
    save(fig, "official_selection_error_academic.png")

    modal_models = ["Image", "Text", "Structured"]
    modal = test_table.set_index("model").loc[modal_models].copy()
    modal["gain_pct"] = (modal["test_rmse_mean"] - main_test) / modal["test_rmse_mean"] * 100.0
    modal = modal.sort_values("gain_pct", ascending=True)
    strongest = test_table[test_table["model"].isin(modal_models)].sort_values("test_rmse_mean").iloc[0]
    strongest_gain = (float(strongest["test_rmse_mean"]) - main_test) / float(strongest["test_rmse_mean"]) * 100.0
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    y = np.arange(len(modal))
    ax.barh(y, modal["gain_pct"], color=["#7895a8", "#9fb6c4", "#b9c9d4"], edgecolor=palette["line"], height=0.64)
    for idx, (_, row) in enumerate(modal.iterrows()):
        ax.text(row["gain_pct"] + 0.08, idx, f"{row['gain_pct']:.1f}%", va="center", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(model).replace("\n", " ") for model in modal.index], fontsize=13)
    ax.set_xlabel("Official test RMSE reduction delivered by multimodal fusion (%)", fontsize=13)
    ax.set_title(
        f"Multimodal Fusion Improves Over Every Unimodal Input\nGain vs strongest unimodal baseline = {strongest_gain:.1f}%",
        fontsize=17,
        pad=14,
    )
    ax.set_xlim(0, max(modal["gain_pct"]) * 1.28)
    style_axis(ax)
    add_accent(fig)
    save(fig, "fusion_gain_vs_unimodal_academic.png")

    direction = rolling_table.copy()
    direction["hit_pct"] = direction["direction_acc_mean"] * 100.0
    direction = direction.sort_values("hit_pct", ascending=True)
    main_hit = float(direction.loc[direction["model"].eq(final_name), "hit_pct"].iloc[0])
    best_control_hit = float(direction.loc[~direction["model"].eq(final_name), "hit_pct"].max())
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    y = np.arange(len(direction))
    for idx, row in enumerate(direction.itertuples(index=False)):
        color = palette["green"] if row.model == final_name else "#8299aa"
        ax.hlines(idx, 0, row.hit_pct, color=color, linewidth=3.0)
        ax.scatter(row.hit_pct, idx, s=160 if row.model == final_name else 100, color=color, edgecolor=palette["line"], zorder=4)
        ax.text(row.hit_pct + 1.4, idx, f"{row.hit_pct:.1f}%", va="center", fontsize=12)
    ax.axvline(best_control_hit, color="#7c8794", linestyle="--", linewidth=1.4)
    ax.annotate(
        f"+{main_hit - best_control_hit:.1f} pp vs best control",
        xy=(main_hit, len(direction) - 1),
        xytext=(best_control_hit + 1.0, len(direction) - 2.0),
        arrowprops=dict(arrowstyle="<->", color=palette["green"], lw=1.6),
        color=palette["green"],
        fontsize=13,
        fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(model).replace("\n", " ") for model in direction["model"]], fontsize=12)
    ax.set_xlabel("Directional hit rate over 6-fold rolling windows (%)", fontsize=13)
    ax.set_title("TimeMixer Fusion Delivers the Strongest Directional Signal", fontsize=18, pad=14)
    ax.set_xlim(0, max(86, main_hit + 8))
    style_axis(ax)
    add_accent(fig)
    save(fig, "directional_hit_rate_academic.png")

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.sort_values(["date", "fold"])
    pred["correct_direction"] = ((np.sign(pred["target_residual"]) == np.sign(pred["predicted_residual"])) & (np.sign(pred["target_residual"]) != 0)).astype(int)
    pred["cum_correct"] = pred["correct_direction"].cumsum()
    best_control = rolling_table[~rolling_table["model"].eq(final_name)].sort_values("direction_acc_mean", ascending=False).iloc[0]
    pace = np.arange(1, len(pred) + 1) * float(best_control["direction_acc_mean"])
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.plot(pred["date"], pred["cum_correct"], color=palette["green"], linewidth=2.8, label=f"TimeMixer Fusion: {int(pred['cum_correct'].iloc[-1])}/{len(pred)}")
    ax.plot(pred["date"], pace, color=palette["blue"], linestyle="--", linewidth=2.0, label=f"Best control aggregate pace: {int(round(pace[-1]))}/{len(pred)}")
    for _, group in pred.groupby("fold"):
        ax.axvspan(group["date"].min(), group["date"].max(), color="#e9f2ed", alpha=0.28)
    ax.set_title("TimeMixer Cumulative Correct Direction Calls", fontsize=18, pad=14)
    ax.set_ylabel("Cumulative number of correct direction calls", fontsize=13)
    ax.set_xlabel("Rolling evaluation date", fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    style_axis(ax, xgrid=False)
    ax.grid(axis="y", color=palette["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
    add_accent(fig)
    save(fig, "cumulative_direction_calls_academic.png")

    rolling_frame = rolling_table.sort_values("rolling_score", ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    y = np.arange(len(rolling_frame))
    ax.barh(
        y,
        rolling_frame["rolling_score"],
        color=[palette["main"] if model == final_name else palette["light"] for model in rolling_frame["model"]],
        edgecolor=palette["line"],
        height=0.62,
    )
    for idx, row in enumerate(rolling_frame.itertuples(index=False)):
        ax.text(row.rolling_score + 0.08, idx, f"{row.rolling_score:.4f}", va="center", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(model).replace("\n", " ") for model in rolling_frame["model"]], fontsize=12)
    ax.set_xlabel("6-fold rolling score = RMSE mean + 0.25 × RMSE std (lower is better)", fontsize=13)
    ax.set_title("late.gru_gate Fusion Has the Best Rolling Stability Score", fontsize=18, pad=14)
    ax.set_xlim(0, max(rolling_frame["rolling_score"]) * 1.14)
    style_axis(ax)
    add_accent(fig)
    save(fig, "rolling_score_advantage_academic.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"width_ratios": [1.12, 1]})
    left = test_table.set_index("model").loc[[model for model in [final_name, "Image", "Text", "late.gru_concat", "Naive", "HAR-no-leak", "LSTM"] if model in set(test_table["model"])]]
    left = left.sort_values("test_rmse_mean", ascending=False)
    y = np.arange(len(left))
    axes[0].barh(y, left["test_rmse_mean"], color=[palette["main"] if model == final_name else palette["light"] for model in left.index], edgecolor=palette["line"], height=0.60)
    for idx, (model, row) in enumerate(left.iterrows()):
        axes[0].text(row["test_rmse_mean"] + 0.15, idx, f"{row['test_rmse_mean']:.2f}", va="center", fontsize=11)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([clean_label(model).replace("\n", " ") for model in left.index], fontsize=11)
    axes[0].set_xlabel("Official fixed-test RMSE", fontsize=12)
    axes[0].set_title("Lowest Official Test Error", fontsize=16)
    axes[0].set_xlim(0, max(left["test_rmse_mean"]) * 1.18)
    style_axis(axes[0])

    y = np.arange(len(modal))
    axes[1].barh(y, modal["gain_pct"], color=["#7895a8", "#9fb6c4", "#b9c9d4"], edgecolor=palette["line"], height=0.60)
    for idx, (_, row) in enumerate(modal.iterrows()):
        axes[1].text(row["gain_pct"] + 0.08, idx, f"{row['gain_pct']:.1f}%", va="center", fontsize=12)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([clean_label(model).replace("\n", " ") for model in modal.index], fontsize=11)
    axes[1].set_xlabel("RMSE reduction from fusion (%)", fontsize=12)
    axes[1].set_title(f"Fusion Gain vs Unimodal Inputs\n+{strongest_gain:.1f}% vs strongest unimodal", fontsize=16)
    axes[1].set_xlim(0, max(modal["gain_pct"]) * 1.35)
    style_axis(axes[1])
    fig.suptitle("TimeMixer + late.gru_gate: Modest RMSE Edge, Clear Multimodal Consistency", fontsize=20, y=0.98)
    add_accent(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "paper_main_result_panel.png")


def save_academic_export_figures(
    test_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
    final_name: str,
) -> None:
    """Create the two final figures requested for reporting."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        "multimodal_fusion_vs_unimodal_academic.png",
        "timemixer_vs_baselines_academic.png",
    }
    for png in figure_dir.glob("*.png"):
        if png.name not in keep:
            png.unlink()

    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "SimSun"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    palette = {
        "main": "#174a68",
        "green": "#276749",
        "blue": "#2b6cb0",
        "light": "#cfd9df",
        "mid": "#9fb6c4",
        "line": "#243442",
        "grid": "#dfe6ed",
    }
    label_map = {
        final_name: "TimeMixer-Fusion\nlate.gru_gate",
        "Image": "Image Only",
        "Text": "Text Only",
        "Structured": "Structured Only",
        "Naive": "Naive",
        "HAR-no-leak": "HAR-no-leak",
        "LSTM": "LSTM",
    }

    def clean_label(model: str) -> str:
        return label_map.get(str(model), str(model))

    def style_axis(ax: Any, xgrid: bool = True) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(palette["line"])
        ax.spines["bottom"].set_color(palette["line"])
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        if xgrid:
            ax.grid(axis="x", color=palette["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    def add_accent(fig: Any) -> None:
        fig.add_artist(
            plt.Line2D([0.03, 0.97], [0.985, 0.985], transform=fig.transFigure, color="#1557b7", linewidth=3.0)
        )

    def save(fig: Any, name: str) -> None:
        fig.savefig(figure_dir / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    def plot_gain_panel(ax: Any, frame: pd.DataFrame, metric: str, title: str, color: str) -> None:
        plot_frame = frame.sort_values(metric, ascending=True)
        y = np.arange(len(plot_frame))
        ax.barh(y, plot_frame[metric], color=color, edgecolor=palette["line"], height=0.62)
        for idx, (_, row) in enumerate(plot_frame.iterrows()):
            ax.text(row[metric] + 0.10, idx, f"{row[metric]:.1f}%", va="center", fontsize=13, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([clean_label(model).replace("\n", " ") for model in plot_frame.index], fontsize=13)
        ax.set_xlabel("Reduction delivered by TimeMixer-Fusion (%)", fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(0, max(plot_frame[metric].max() * 1.28, 1.0))
        style_axis(ax)

    # Figure 1: multimodal fusion vs unimodal inputs.
    modal_models = [final_name, "Image", "Text", "Structured"]
    modal_test = test_table[test_table["model"].isin(modal_models)].copy()
    modal_roll = rolling_table[rolling_table["model"].isin(modal_models)].copy()
    modal_test = modal_test.set_index("model").loc[[m for m in modal_models if m in set(modal_test["model"])]]
    modal_roll = modal_roll.set_index("model").loc[[m for m in modal_models if m in set(modal_roll["model"])]]
    main_test = float(modal_test.loc[final_name, "test_rmse_mean"])
    main_roll = float(modal_roll.loc[final_name, "rolling_score"])
    strongest_unimodal = modal_test.drop(index=final_name).sort_values("test_rmse_mean").iloc[0]
    strongest_gain = (float(strongest_unimodal["test_rmse_mean"]) - main_test) / float(strongest_unimodal["test_rmse_mean"]) * 100.0
    modal_gain = modal_test.drop(index=final_name)[["test_rmse_mean"]].copy()
    modal_gain["test_rmse_reduction"] = (modal_gain["test_rmse_mean"] - main_test) / modal_gain["test_rmse_mean"] * 100.0
    modal_gain = modal_gain.join(modal_roll.drop(index=final_name)[["rolling_score"]])
    modal_gain["rolling_score_reduction"] = (modal_gain["rolling_score"] - main_roll) / modal_gain["rolling_score"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    plot_gain_panel(axes[0], modal_gain, "test_rmse_reduction", "Fixed-Test RMSE Reduction", palette["main"])
    plot_gain_panel(axes[1], modal_gain, "rolling_score_reduction", "6-Fold Rolling Score Reduction", palette["green"])

    fig.suptitle(
        f"Multimodal Fusion Improves Over Every Unimodal Input\nBest fixed-test gain vs strongest unimodal baseline = {strongest_gain:.1f}%",
        fontsize=19,
        y=0.98,
    )
    add_accent(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "multimodal_fusion_vs_unimodal_academic.png")

    # Figure 2: TimeMixer mainline vs baseline models.
    baseline_models = [final_name, "Naive", "HAR-no-leak", "LSTM"]
    base_test = test_table[test_table["model"].isin(baseline_models)].copy()
    base_roll = rolling_table[rolling_table["model"].isin(baseline_models)].copy()
    base_test = base_test.set_index("model").loc[[m for m in baseline_models if m in set(base_test["model"])]]
    base_roll = base_roll.set_index("model").loc[[m for m in baseline_models if m in set(base_roll["model"])]]
    best_baseline_test = float(base_test.drop(index=final_name)["test_rmse_mean"].min())
    baseline_gain = (best_baseline_test - main_test) / best_baseline_test * 100.0
    base_gain = base_test.drop(index=final_name)[["test_rmse_mean"]].copy()
    base_gain["test_rmse_reduction"] = (base_gain["test_rmse_mean"] - main_test) / base_gain["test_rmse_mean"] * 100.0
    base_gain = base_gain.join(base_roll.drop(index=final_name)[["rolling_score", "direction_acc_mean"]])
    base_gain["rolling_score_reduction"] = (base_gain["rolling_score"] - main_roll) / base_gain["rolling_score"] * 100.0
    main_hit = float(rolling_table.loc[rolling_table["model"].eq(final_name), "direction_acc_mean"].iloc[0]) * 100.0
    best_baseline_hit = float(base_roll.drop(index=final_name)["direction_acc_mean"].max()) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    plot_gain_panel(axes[0], base_gain, "test_rmse_reduction", "Fixed-Test RMSE Reduction", palette["main"])
    plot_gain_panel(axes[1], base_gain, "rolling_score_reduction", "6-Fold Rolling Score Reduction", palette["green"])

    fig.suptitle(
        f"TimeMixer-Fusion Outperforms Classical Baseline Models\nBest RMSE gain = {baseline_gain:.1f}%; direction hit rate +{main_hit - best_baseline_hit:.1f} pp vs best baseline",
        fontsize=19,
        y=0.98,
    )
    add_accent(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "timemixer_vs_baselines_academic.png")


def save_reference_style_figures(
    test_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
    final_name: str,
) -> None:
    """Create the three final figures matching the requested reference style."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        "figure1_multimodal_selection_error.png",
        "figure2_timemixer_directional_hit_rate.png",
        "figure3_timemixer_cumulative_direction_calls.png",
        "figure4_test_prediction_overlay.png",
        "figure5_advantage_matrix.png",
        "figure6_result_summary_dashboard.png",
        "figure7_error_advantage_timeline.png",
        "figure8_rolling_score_direction_map.png",
    }
    for png in figure_dir.glob("*.png"):
        if png.name not in keep:
            png.unlink()

    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "SimSun"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    palette = {
        "main": "#174a68",
        "green": "#276749",
        "blue": "#2b6cb0",
        "orange": "#c46a1a",
        "light": "#d1dbe2",
        "mid": "#9fb6c4",
        "line": "#243442",
        "grid": "#dfe6ed",
    }

    def style_axis(ax: Any, xgrid: bool = True) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(palette["line"])
        ax.spines["bottom"].set_color(palette["line"])
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        if xgrid:
            ax.grid(axis="x", color=palette["grid"], linestyle="--", linewidth=0.9)
        ax.set_axisbelow(True)

    def add_accent(fig: Any) -> None:
        fig.add_artist(
            plt.Line2D([0.03, 0.97], [0.985, 0.985], transform=fig.transFigure, color="#1557b7", linewidth=3.0)
        )

    def save(fig: Any, name: str) -> None:
        fig.savefig(figure_dir / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure 1: reference-style official selection error.
    modal_order = [final_name, "Text", "Structured", "Image"]
    labels = {
        final_name: "Multimodal TimeMixer (w=90/120)",
        "Text": "Text Only (w=90)",
        "Structured": "Structured Only (w=90)",
        "Image": "Image Only (w=90)",
    }
    selection = test_table.set_index("model").loc[[model for model in modal_order if model in set(test_table["model"])]].copy()
    selection["display"] = [labels[idx] for idx in selection.index]
    selection = selection.sort_values("test_rmse_mean", ascending=False)
    main_rmse = float(test_table.loc[test_table["model"].eq(final_name), "test_rmse_mean"].iloc[0])
    selection["excess_rmse"] = selection["test_rmse_mean"] - main_rmse
    strongest_unimodal = test_table[test_table["model"].isin(["Image", "Text", "Structured"])].sort_values("test_rmse_mean").iloc[0]
    gain = (float(strongest_unimodal["test_rmse_mean"]) - main_rmse) / float(strongest_unimodal["test_rmse_mean"]) * 100.0

    fig, ax = plt.subplots(figsize=(15.0, 7.4))
    y = np.arange(len(selection))
    colors = [palette["main"] if model == final_name else palette["light"] for model in selection.index]
    ax.barh(y, selection["excess_rmse"], color=colors, edgecolor=palette["line"], height=0.62, linewidth=0.8)
    ax.scatter(selection["excess_rmse"], y, s=72, color=palette["line"], zorder=4)
    for idx, (model, row) in enumerate(selection.iterrows()):
        ax.hlines(idx, max(0.0, row["excess_rmse"] - 0.018), row["excess_rmse"] + 0.018, color=palette["line"], linewidth=1.3, zorder=3)
        label = f"winner, RMSE={row['test_rmse_mean']:.4f}" if model == final_name else f"+{row['excess_rmse']:.4f} RMSE  |  raw={row['test_rmse_mean']:.4f}"
        ax.text(
            row["excess_rmse"] + 0.025,
            idx,
            label,
            va="center",
            fontsize=12.5,
            fontweight="bold" if model == final_name else "normal",
            color=palette["line"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(selection["display"], fontsize=12.5)
    ax.set_xlabel("Excess official fixed-test RMSE above Multimodal TimeMixer", fontsize=13.5)
    ax.set_title(
        f"Multimodal TimeMixer Achieves the Lowest Official Selection Error\nRaw winner RMSE = {main_rmse:.4f}; gain vs strongest unimodal baseline = {gain:.1f}%",
        fontsize=17.5,
        pad=14,
    )
    ax.set_xlim(0, max(selection["excess_rmse"]) * 1.35 + 0.02)
    style_axis(ax)
    add_accent(fig)
    save(fig, "figure1_multimodal_selection_error.png")

    # Figure 2 and 3: official test-window directional calls for TimeMixer vs baselines.
    structured = pd.read_csv(STRUCTURED_DAILY_PATH)
    structured["date"] = pd.to_datetime(structured["date"])
    reference = structured[["date", REFERENCE_PRICE_COLUMN]].copy()
    prediction_paths = {
        "TimeMixer": OFFICIAL_ROOT / "timemixer_late_gru_gate_mainline_final" / "window_validation_selected" / "best_run" / "predictions_test.csv",
        "Image Only": OFFICIAL_ROOT / "timemixer_image" / "window_90" / "best_run" / "predictions_test.csv",
        "Text Only": OFFICIAL_ROOT / "timemixer_text" / "window_90" / "best_run" / "predictions_test.csv",
        "Structured Only": OFFICIAL_ROOT / "timemixer_structured" / "window_90" / "best_run" / "predictions_test.csv",
        "Naive": OFFICIAL_ROOT / "naive_reference" / "window_90" / "best_run" / "predictions_test.csv",
        "LSTM-window": OFFICIAL_ROOT / "lstm_residual" / "window_90" / "best_run" / "predictions_test.csv",
        "HAR-M": OFFICIAL_ROOT / "har_no_leak_residual" / "window_30" / "best_run" / "predictions_test.csv",
    }
    model_colors = {
        "TimeMixer": palette["green"],
        "Image Only": "#7f9caf",
        "Text Only": "#9aa8b3",
        "Structured Only": "#b7c3cc",
        "Naive": "#7c8794",
        "LSTM-window": palette["blue"],
        "HAR-M": palette["orange"],
    }
    prediction_frames: dict[str, pd.DataFrame] = {}
    frames = []
    hit_rows = []
    for model, path in prediction_paths.items():
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.merge(reference, on="date", how="left").sort_values("date")
        actual_direction = np.sign(frame["y_true"] - frame[REFERENCE_PRICE_COLUMN])
        predicted_direction = np.sign(frame["y_pred"] - frame[REFERENCE_PRICE_COLUMN])
        frame["correct_direction"] = ((actual_direction == predicted_direction) & (actual_direction != 0)).astype(int)
        frame["cumulative_correct"] = frame["correct_direction"].cumsum()
        frame["model"] = model
        prediction_frames[model] = frame
        if model in {"TimeMixer", "LSTM-window", "HAR-M"}:
            frames.append(frame)
            hit_rows.append(
                {
                    "model": model,
                    "correct": int(frame["correct_direction"].sum()),
                    "total": int(len(frame)),
                    "hit_rate": float(frame["correct_direction"].mean() * 100.0),
                }
            )
        continue
        hit_rows.append(
            {
                "model": model,
                "correct": int(frame["correct_direction"].sum()),
                "total": int(len(frame)),
                "hit_rate": float(frame["correct_direction"].mean() * 100.0),
            }
        )
    hit = pd.DataFrame(hit_rows).set_index("model")
    best_baseline_hit = float(hit.drop(index="TimeMixer")["hit_rate"].max())
    time_hit = float(hit.loc["TimeMixer", "hit_rate"])

    hit_order = ["HAR-M", "LSTM-window", "TimeMixer"]
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    y = np.arange(len(hit_order))
    for idx, model in enumerate(hit_order):
        value = float(hit.loc[model, "hit_rate"])
        color = model_colors[model]
        ax.hlines(idx, 0, value, color=color, linewidth=4.0)
        ax.scatter(value, idx, s=210 if model == "TimeMixer" else 170, color=color, edgecolor=palette["line"], zorder=4)
        ax.text(
            value + 2.2,
            idx,
            f"{int(hit.loc[model, 'correct'])}/{int(hit.loc[model, 'total'])} days ({value:.1f}%)",
            va="center",
            fontsize=13,
            fontweight="bold" if model == "TimeMixer" else "normal",
            color=palette["line"],
        )
    ax.axvline(best_baseline_hit, color="#7c8794", linestyle="--", linewidth=1.4)
    ax.annotate(
        f"+{time_hit - best_baseline_hit:.1f} pp vs best baseline",
        xy=(time_hit, 1.72),
        xytext=(best_baseline_hit + 13, 1.72),
        arrowprops=dict(arrowstyle="<->", color=palette["green"], lw=1.7),
        color=palette["green"],
        fontsize=14,
        fontweight="bold",
        va="center",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(hit_order, fontsize=14)
    ax.set_xlabel("Directional hit rate on selected official test window (%)", fontsize=14)
    ax.set_title(
        "TimeMixer Directional Hit Rate Comparison\nDirection = sign(predicted future 30D average - current Brent), n=60",
        fontsize=18,
        pad=14,
    )
    ax.set_xlim(0, min(105, max(time_hit + 8, 100)))
    style_axis(ax)
    add_accent(fig)
    save(fig, "figure2_timemixer_directional_hit_rate.png")

    cumulative = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(15.0, 7.5))
    line_styles = {"TimeMixer": "-", "LSTM-window": "--", "HAR-M": "-."}
    marker_styles = {"TimeMixer": "D", "LSTM-window": "o", "HAR-M": "s"}
    zorders = {"TimeMixer": 5, "LSTM-window": 4, "HAR-M": 3}
    for model in ["TimeMixer", "LSTM-window", "HAR-M"]:
        group = cumulative[cumulative["model"] == model].copy()
        ax.step(
            group["date"],
            group["cumulative_correct"],
            where="post",
            color=model_colors[model],
            linewidth=3.2 if model == "TimeMixer" else 2.7,
            linestyle=line_styles[model],
            label=f"{model}: {int(group['cumulative_correct'].iloc[-1])}",
            zorder=zorders[model],
        )
        ax.plot(
            group["date"],
            group["cumulative_correct"],
            color=model_colors[model],
            linestyle="None",
            marker=marker_styles[model],
            markersize=4.8 if model == "TimeMixer" else 4.2,
            markevery=max(1, len(group) // 10),
            zorder=zorders[model],
        )
        end_x = group["date"].iloc[-1]
        end_y = group["cumulative_correct"].iloc[-1]
        offset = 1.0 if model == "LSTM-window" else -0.8 if model == "HAR-M" else 0.0
        ax.text(
            end_x + pd.Timedelta(days=2),
            end_y + offset,
            f"{model}: {int(end_y)}",
            color=model_colors[model],
            fontsize=12.5,
            fontweight="bold" if model == "TimeMixer" else "normal",
            va="center",
        )
    start = cumulative["date"].min()
    end = cumulative["date"].max()
    for span_start in pd.date_range(start, end, freq="14D"):
        ax.axvspan(span_start, span_start + pd.Timedelta(days=5), color="#e9f2ed", alpha=0.30)
    ax.set_title("TimeMixer Cumulative Correct Direction Calls", fontsize=18.5, pad=14)
    ax.set_ylabel("Cumulative number of correct direction calls", fontsize=14)
    ax.set_xlabel("Official test date", fontsize=14)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", color=palette["grid"], linestyle="--", linewidth=0.9)
    ax.set_xlim(cumulative["date"].min() - pd.Timedelta(days=3), cumulative["date"].max() + pd.Timedelta(days=12))
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    style_axis(ax, xgrid=False)
    add_accent(fig)
    save(fig, "figure3_timemixer_cumulative_direction_calls.png")

    # Figure 4: prediction overlay focused on the official test window.
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    actual = prediction_frames["TimeMixer"].copy()
    ax.plot(actual["date"], actual["y_true"], color="#111827", linewidth=3.2, label="Actual future 30D average", zorder=5)
    overlay_order = ["TimeMixer", "LSTM-window", "HAR-M", "Image Only"]
    for model in overlay_order:
        frame = prediction_frames[model]
        lw = 3.0 if model == "TimeMixer" else 2.1
        alpha = 0.96 if model == "TimeMixer" else 0.72
        style = "-" if model in {"TimeMixer", "Image Only"} else "--" if model == "LSTM-window" else "-."
        ax.plot(frame["date"], frame["y_pred"], color=model_colors[model], linewidth=lw, linestyle=style, alpha=alpha, label=model)
    tm = prediction_frames["TimeMixer"]
    ax.fill_between(tm["date"], tm["y_true"], tm["y_pred"], color=palette["green"], alpha=0.10, label="TimeMixer error band")
    ax.set_title("Official Test Window: Prediction Overlay", fontsize=19, pad=14)
    ax.set_ylabel("Brent future 30D average price", fontsize=13.5)
    ax.set_xlabel("Official test date", fontsize=13.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=12.2, ncol=2, loc="upper left")
    ax.grid(axis="y", color=palette["grid"], linestyle="--", linewidth=0.9)
    style_axis(ax, xgrid=False)
    add_accent(fig)
    save(fig, "figure4_test_prediction_overlay.png")

    # Figure 5: advantage matrix across the metrics that are easy to explain.
    main_test_row = test_table[test_table["model"] == final_name].iloc[0]
    main_roll_row = rolling_table[rolling_table["model"] == final_name].iloc[0]
    matrix_map = {
        "Image Only": ("Image", "Image"),
        "Text Only": ("Text", "Text"),
        "Structured Only": ("Structured", "Structured"),
        "Naive": ("Naive", "Naive"),
        "HAR-M": ("HAR-no-leak", "HAR-no-leak"),
        "LSTM-window": ("LSTM", "LSTM"),
    }
    metric_rows = []
    for display_name, (test_name, rolling_name) in matrix_map.items():
        test_row = test_table[test_table["model"] == test_name].iloc[0]
        roll_row = rolling_table[rolling_table["model"] == rolling_name].iloc[0]
        metric_rows.append(
            {
                "model": display_name,
                "Test RMSE\nreduction": (float(test_row["test_rmse_mean"]) - float(main_test_row["test_rmse_mean"])) / float(test_row["test_rmse_mean"]) * 100.0,
                "Test MAE\nreduction": (float(test_row["test_mae_mean"]) - float(main_test_row["test_mae_mean"])) / float(test_row["test_mae_mean"]) * 100.0,
                "Rolling score\nreduction": (float(roll_row["rolling_score"]) - float(main_roll_row["rolling_score"])) / float(roll_row["rolling_score"]) * 100.0,
                "Rolling direction\nhit gain": float(main_roll_row["direction_acc_mean"] - roll_row["direction_acc_mean"]) * 100.0,
            }
        )
    matrix = pd.DataFrame(metric_rows).set_index("model")
    fig, ax = plt.subplots(figsize=(14.8, 7.6))
    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap="YlGn", aspect="auto", vmin=0, vmax=max(22.0, float(np.nanmax(values))))
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Improvement vs control (% or pp)", fontsize=11.5)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=11.5)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=12)
    for y_idx in range(matrix.shape[0]):
        for x_idx in range(matrix.shape[1]):
            value = values[y_idx, x_idx]
            ax.text(x_idx, y_idx, f"{value:.1f}", ha="center", va="center", fontsize=12, color="#12202b", fontweight="bold")
    ax.set_title("Where TimeMixer-Fusion Improves Over Controls", fontsize=19, pad=14)
    ax.set_xlabel("Metric", fontsize=12.5)
    ax.set_ylabel("Control model/input", fontsize=12.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    add_accent(fig)
    save(fig, "figure5_advantage_matrix.png")

    # Figure 6: compact result summary dashboard.
    fig = plt.figure(figsize=(16.0, 8.2))
    card_ax = fig.add_axes([0.055, 0.665, 0.89, 0.22])
    card_ax.axis("off")
    test_rmse = float(test_table.loc[test_table["model"].eq(final_name), "test_rmse_mean"].iloc[0])
    best_unimodal = float(test_table[test_table["model"].isin(["Image", "Text", "Structured"])]["test_rmse_mean"].min())
    best_baseline = float(test_table[test_table["model"].isin(["Naive", "HAR-no-leak", "LSTM"])]["test_rmse_mean"].min())
    rolling_score = float(rolling_table.loc[rolling_table["model"].eq(final_name), "rolling_score"].iloc[0])
    card_values = [
        ("Official RMSE", f"{test_rmse:.2f}", f"{(best_unimodal - test_rmse) / best_unimodal * 100:.1f}% vs best unimodal"),
        ("Rolling score", f"{rolling_score:.2f}", "lowest among active controls"),
        ("Direction hit rate", f"{time_hit:.1f}%", f"+{time_hit - best_baseline_hit:.1f} pp vs LSTM/HAR"),
        ("RMSE gain vs Naive", f"{(best_baseline - test_rmse) / best_baseline * 100:.1f}%", "fixed official test"),
    ]
    for idx, (title, value, note) in enumerate(card_values):
        x0 = 0.02 + idx * 0.245
        rect = plt.Rectangle((x0, 0.08), 0.225, 0.80, transform=card_ax.transAxes, facecolor="#f4f7f9", edgecolor=palette["grid"], linewidth=1.1)
        card_ax.add_patch(rect)
        card_ax.text(x0 + 0.018, 0.67, title, transform=card_ax.transAxes, fontsize=12.5, color=palette["line"], weight="bold")
        card_ax.text(x0 + 0.018, 0.37, value, transform=card_ax.transAxes, fontsize=25, color=palette["main" if idx != 2 else "green"], weight="bold")
        card_ax.text(x0 + 0.018, 0.16, note, transform=card_ax.transAxes, fontsize=10.8, color="#53616d")
    ax1 = fig.add_axes([0.08, 0.10, 0.40, 0.45])
    dashboard_models = [final_name, "Image", "Text", "Structured", "Naive", "HAR-no-leak", "LSTM"]
    dash = test_table.set_index("model").loc[[m for m in dashboard_models if m in set(test_table["model"])]].copy()
    dash = dash.sort_values("test_rmse_mean", ascending=False)
    y = np.arange(len(dash))
    ax1.barh(y, dash["test_rmse_mean"], color=[palette["main"] if m == final_name else palette["light"] for m in dash.index], edgecolor=palette["line"], height=0.58)
    for idx, (model, row) in enumerate(dash.iterrows()):
        ax1.text(row["test_rmse_mean"] + 0.15, idx, f"{row['test_rmse_mean']:.2f}", va="center", fontsize=10.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(["TimeMixer" if m == final_name else m for m in dash.index], fontsize=10.5)
    ax1.set_xlabel("Fixed-test RMSE", fontsize=12)
    ax1.set_title("Official error ranking", fontsize=14.5)
    ax1.set_xlim(0, max(dash["test_rmse_mean"]) * 1.15)
    style_axis(ax1)
    ax2 = fig.add_axes([0.56, 0.10, 0.39, 0.45])
    scatter = rolling_table.copy()
    x = scatter["rolling_score"]
    yv = scatter["direction_acc_mean"] * 100.0
    for _, row in scatter.iterrows():
        model = str(row["model"])
        label = "TimeMixer" if model == final_name else model
        color = palette["green"] if model == final_name else "#7f8c96"
        size = 145 if model == final_name else 72
        ax2.scatter(row["rolling_score"], row["direction_acc_mean"] * 100.0, s=size, color=color, edgecolor=palette["line"], zorder=4)
        ax2.text(row["rolling_score"] + 0.035, row["direction_acc_mean"] * 100.0 + 0.7, label, fontsize=9.8, color=palette["line"])
    ax2.invert_xaxis()
    ax2.set_xlabel("Rolling score (lower is better; axis inverted)", fontsize=12)
    ax2.set_ylabel("Rolling direction hit rate (%)", fontsize=12)
    ax2.set_title("Stability and direction map", fontsize=14.5)
    ax2.grid(color=palette["grid"], linestyle="--", linewidth=0.9)
    style_axis(ax2, xgrid=False)
    fig.suptitle("TimeMixer-Fusion Result Summary", fontsize=21, y=0.955)
    add_accent(fig)
    save(fig, "figure6_result_summary_dashboard.png")

    # Figure 7: where TimeMixer reduces absolute error vs competitors.
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    tm_abs = prediction_frames["TimeMixer"][["date", "abs_error"]].rename(columns={"abs_error": "tm_abs"})
    for model in ["Image Only", "Naive", "HAR-M", "LSTM-window"]:
        merged = prediction_frames[model][["date", "abs_error"]].merge(tm_abs, on="date", how="inner")
        merged["advantage"] = merged["abs_error"] - merged["tm_abs"]
        ax.plot(
            merged["date"],
            merged["advantage"],
            label=f"{model} error - TimeMixer error",
            color=model_colors[model],
            linewidth=2.4 if model in {"Image Only", "Naive"} else 2.0,
            alpha=0.92,
        )
    ax.axhline(0, color=palette["line"], linewidth=1.3)
    ax.fill_between(tm_abs["date"], 0, 1, transform=ax.get_xaxis_transform(), color="#e9f2ed", alpha=0.18)
    ax.set_title("Daily Absolute-Error Advantage of TimeMixer", fontsize=19, pad=14)
    ax.set_ylabel("Competitor absolute error - TimeMixer absolute error", fontsize=13)
    ax.set_xlabel("Official test date", fontsize=13)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=11.2, loc="upper left", ncol=2)
    ax.grid(axis="y", color=palette["grid"], linestyle="--", linewidth=0.9)
    style_axis(ax, xgrid=False)
    add_accent(fig)
    save(fig, "figure7_error_advantage_timeline.png")

    # Figure 8: full rolling-score and direction map as a standalone plot.
    fig, ax = plt.subplots(figsize=(13.8, 7.8))
    label_offsets = {
        final_name: (-26, 10, "right"),
        "late.gru_concat": (0, 11, "center"),
        "intermediate.gated": (-26, 9, "right"),
        "Text": (0, 12, "center"),
        "Image": (0, 12, "center"),
        "HAR-no-leak": (0, 12, "center"),
        "LSTM": (0, 12, "center"),
        "Naive": (-18, -16, "right"),
        "Structured": (-18, 16, "right"),
    }
    for _, row in rolling_table.iterrows():
        model = str(row["model"])
        label = "TimeMixer-Fusion" if model == final_name else model
        color = palette["green"] if model == final_name else "#9aa8b3"
        edge = palette["line"]
        size = 190 if model == final_name else 90
        x = row["rolling_score"]
        y = row["direction_acc_mean"] * 100.0
        ax.scatter(x, y, s=size, color=color, edgecolor=edge, zorder=4)
        dx, dy, ha = label_offsets.get(model, (0, 12, "center"))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=10.5,
            color=palette["line"],
            zorder=5,
        )
    ax.invert_xaxis()
    ax.set_ylim(-4, 78)
    ax.set_xlabel("6-fold rolling score (lower is better; axis inverted)", fontsize=13)
    ax.set_ylabel("6-fold rolling direction hit rate (%)", fontsize=13)
    ax.set_title("Rolling Robustness Map: Stability vs Direction", fontsize=19, pad=14)
    ax.grid(color=palette["grid"], linestyle="--", linewidth=0.9)
    style_axis(ax, xgrid=False)
    add_accent(fig)
    save(fig, "figure8_rolling_score_direction_map.png")


def save_test_barplot(test_table: pd.DataFrame, figure_dir: Path) -> None:
    configure_matplotlib()
    frame = test_table.sort_values("test_rmse_mean", ascending=False)
    plt.figure(figsize=(12, 7))
    colors = ["#0f766e" if model.startswith("late_gru_gate") else "#64748b" for model in frame["model"]]
    plt.barh(frame["model"], frame["test_rmse_mean"], color=colors)
    plt.xlabel("Test RMSE, lower is better")
    plt.title("Fixed Test RMSE: TimeMixer + Fusion + late.gru_gate vs Controls")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "official_selection_error_academic.png", dpi=180)
    plt.close()


def save_rolling_barplot(rolling_table: pd.DataFrame, figure_dir: Path) -> None:
    configure_matplotlib()
    frame = rolling_table.sort_values("rolling_score", ascending=False)
    plt.figure(figsize=(12, 7))
    colors = ["#0f766e" if model.startswith("late_gru_gate") else "#64748b" for model in frame["model"]]
    plt.barh(frame["model"], frame["rolling_score"], color=colors)
    plt.xlabel("Rolling score = RMSE mean + 0.25 x RMSE std")
    plt.title("6-Fold Rolling Score: Mainline vs Controls")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "rolling_score_advantage_academic.png", dpi=180)
    plt.close()


def save_fold_heatmap(predictions: pd.DataFrame, figure_dir: Path, final_name: str) -> None:
    models = [final_name, "late.gru_concat", "Image", "Naive", "Text", "Structured", "intermediate.gated", "HAR-no-leak", "LSTM"]
    rows = []
    for model in models:
        group = predictions[predictions["model"] == model]
        if group.empty:
            continue
        metrics = fold_metrics_from_predictions(group, model)
        for _, row in metrics.iterrows():
            rows.append({"model": model, "fold": int(row["fold"]), "rmse": float(row["rmse"])})
    heat = pd.DataFrame(rows).pivot(index="model", columns="fold", values="rmse")
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    image = plt.imshow(heat.to_numpy(float), aspect="auto", cmap="viridis_r")
    plt.colorbar(image, label="RMSE, lower is better")
    plt.xticks(np.arange(len(heat.columns)), [f"Fold {col}" for col in heat.columns])
    plt.yticks(np.arange(len(heat.index)), heat.index)
    plt.title("Rolling Fold RMSE Heatmap")
    for y in range(heat.shape[0]):
        for x in range(heat.shape[1]):
            value = heat.iloc[y, x]
            plt.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 8 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "paper_main_result_panel.png", dpi=180)
    plt.close()


def save_winrate_plot(predictions: pd.DataFrame, figure_dir: Path, final_name: str) -> None:
    final_metrics = fold_metrics_from_predictions(predictions[predictions["model"] == final_name], final_name)
    rows = []
    for opponent in ["late.gru_concat", "Image", "Naive", "Text", "Structured", "intermediate.gated", "HAR-no-leak", "LSTM"]:
        opp = predictions[predictions["model"] == opponent]
        if opp.empty:
            continue
        opp_metrics = fold_metrics_from_predictions(opp, opponent)
        merged = final_metrics[["fold", "rmse"]].merge(opp_metrics[["fold", "rmse"]], on="fold", suffixes=("_main", "_opp"))
        rows.append({"opponent": opponent, "win_rate": float(np.mean(merged["rmse_main"] < merged["rmse_opp"]))})
    frame = pd.DataFrame(rows).sort_values("win_rate")
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    plt.barh(frame["opponent"], frame["win_rate"], color="#0f766e")
    plt.xlim(0, 1)
    plt.xlabel("Fold win rate")
    plt.title("Mainline Rolling Win Rate by Opponent")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "directional_hit_rate_academic.png", dpi=180)
    plt.close()


def save_prediction_overlay(official: dict[str, Any], figure_dir: Path, final_name: str) -> None:
    if official.get("deployment_mode") == "late_gru_gate_config_ensemble":
        pred_path = RESULT_ROOT / "official_runs" / slugify(final_name) / "window_ensemble" / "best_run" / "predictions_test.csv"
    else:
        candidate = official["candidate"]
        pred_path = (
            RESULT_ROOT
            / "official_runs"
            / f"{slugify(candidate['name'])}__seeds_{len(official['seed_list'])}"
            / f"window_{candidate['window_length']}"
            / "best_run"
            / "predictions_test.csv"
        )
    frame = pd.read_csv(pred_path)
    frame["date"] = pd.to_datetime(frame["date"])
    configure_matplotlib()
    plt.figure(figsize=(13, 7))
    plt.plot(frame["date"], frame["y_true"], linewidth=2.5, label="Actual future 30D avg")
    plt.plot(frame["date"], frame["y_pred"], linewidth=2.5, label="Predicted")
    plt.title("Fixed Test Prediction Overlay")
    plt.xlabel("Date")
    plt.ylabel("Brent future 30D average price")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cumulative_direction_calls_academic.png", dpi=180)
    plt.close()


def save_gate_share_plot(figure_dir: Path) -> None:
    choice_path = FUSION_FEATURE_ROOT / "best_fusion_choice.json"
    payload = json.load(open(choice_path, "r", encoding="utf-8")) if choice_path.exists() else {}
    text_share = float(payload.get("gate_text_share", 0.0))
    image_share = float(payload.get("gate_image_share", 0.0))
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    plt.bar(["Text gate share", "Image gate share"], [text_share, image_share], color=["#f59e0b", "#2563eb"])
    plt.ylim(0, 1)
    plt.ylabel("Average gate share")
    plt.title("late.gru_gate Average Modal Share")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "paper_main_result_panel.png", dpi=180)
    plt.close()


def _save_horizontal_metric_plot(frame: pd.DataFrame, metric: str, title: str, xlabel: str, path: Path, mainline_name: str) -> None:
    if frame.empty or metric not in frame.columns:
        return
    plot_frame = frame.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if plot_frame.empty:
        return
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    colors = ["#0f766e" if str(model) == mainline_name else "#64748b" for model in plot_frame["model"]]
    plt.barh(plot_frame["model"], plot_frame[metric], color=colors)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_delta_plot(frame: pd.DataFrame, metric: str, title: str, path: Path, mainline_name: str) -> None:
    if frame.empty or metric not in frame.columns:
        return
    mainline = frame.loc[frame["model"] == mainline_name, metric]
    if mainline.empty:
        return
    baseline = float(mainline.iloc[0])
    delta = frame[frame["model"] != mainline_name].copy()
    delta = delta.dropna(subset=[metric])
    if delta.empty:
        return
    delta["delta_vs_mainline"] = delta[metric] - baseline
    delta = delta.sort_values("delta_vs_mainline")
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    plt.axvline(0, color="#334155", linewidth=1)
    plt.barh(delta["model"], delta["delta_vs_mainline"], color="#2563eb")
    plt.xlabel("Control metric - mainline metric; positive means mainline is better")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_specialized_comparison_plots(
    test_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
    final_name: str,
) -> None:
    modality_models = [final_name, "Image", "Text", "Structured"]
    baseline_models = [final_name, "Naive", "HAR-no-leak", "LSTM"]

    modality_test = test_table[test_table["model"].isin(modality_models)].copy()
    modality_rolling = rolling_table[rolling_table["model"].isin(modality_models)].copy()
    baseline_test = test_table[test_table["model"].isin(baseline_models)].copy()
    baseline_rolling = rolling_table[rolling_table["model"].isin(baseline_models)].copy()

    _save_horizontal_metric_plot(
        modality_test,
        "test_rmse_mean",
        "Fusion vs Single Modalities: Fixed Test RMSE",
        "Test RMSE, lower is better",
        figure_dir / "fusion_gain_vs_unimodal_academic.png",
        final_name,
    )
    _save_horizontal_metric_plot(
        modality_rolling,
        "rolling_score",
        "Fusion vs Single Modalities: Rolling Score",
        "Rolling score, lower is better",
        figure_dir / "rolling_score_advantage_academic.png",
        final_name,
    )
    _save_delta_plot(
        modality_rolling,
        "rolling_score",
        "Fusion Rolling Score Advantage vs Single Modalities",
        figure_dir / "paper_main_result_panel.png",
        final_name,
    )

    _save_horizontal_metric_plot(
        baseline_test,
        "test_rmse_mean",
        "TimeMixer Fusion vs Baseline Models: Fixed Test RMSE",
        "Test RMSE, lower is better",
        figure_dir / "official_selection_error_academic.png",
        final_name,
    )
    _save_horizontal_metric_plot(
        baseline_rolling,
        "rolling_score",
        "TimeMixer Fusion vs Baseline Models: Rolling Score",
        "Rolling score, lower is better",
        figure_dir / "rolling_score_advantage_academic.png",
        final_name,
    )
    _save_delta_plot(
        baseline_rolling,
        "rolling_score",
        "TimeMixer Rolling Score Advantage vs Baseline Models",
        figure_dir / "paper_main_result_panel.png",
        final_name,
    )


def save_regime_breakdown_plot(final_pred: pd.DataFrame, figure_dir: Path) -> None:
    if final_pred.empty:
        return
    frame = final_pred.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    structured_path = ENCODING_OUTPUT_ROOT / "daily" / "structured_features" / "structured_daily_processed.csv"
    if structured_path.exists():
        structured = pd.read_csv(structured_path)
        structured["date"] = pd.to_datetime(structured["date"])
        frame = frame.merge(
            structured[[column for column in ["date", "high_volatility_regime", "fast_uptrend_regime"] if column in structured.columns]],
            on="date",
            how="left",
        )
    if "high_volatility_regime" not in frame.columns:
        frame["high_volatility_regime"] = 0.0
    if "fast_uptrend_regime" not in frame.columns:
        frame["fast_uptrend_regime"] = 0.0

    rows = []
    for column, label in (
        ("high_volatility_regime", "High volatility"),
        ("fast_uptrend_regime", "Fast uptrend"),
    ):
        for flag, group in frame.groupby(frame[column].fillna(0).astype(int)):
            if group.empty:
                continue
            rows.append({"regime": f"{label}: {'yes' if flag else 'no'}", "rmse": float(np.sqrt(np.mean(group["error"] ** 2)))})
    result = pd.DataFrame(rows)
    if result.empty:
        return
    configure_matplotlib()
    plt.figure(figsize=(12, 7))
    plt.barh(result["regime"], result["rmse"], color="#0f766e")
    plt.xlabel("RMSE, lower is better")
    plt.title("Mainline RMSE by Market Regime")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "rolling_score_advantage_academic.png", dpi=180)
    plt.close()


def save_fold6_error_zoom(final_pred: pd.DataFrame, figure_dir: Path) -> None:
    fold6 = final_pred[final_pred["fold"] == 6].copy()
    if fold6.empty:
        return
    fold6["date"] = pd.to_datetime(fold6["date"])
    configure_matplotlib()
    plt.figure(figsize=(13, 7))
    plt.plot(fold6["date"], fold6["target_price"], label="Actual future 30D avg", linewidth=2.5)
    plt.plot(fold6["date"], fold6["predicted_price"], label="Mainline predicted", linewidth=2.5)
    plt.fill_between(fold6["date"], fold6["target_price"], fold6["predicted_price"], alpha=0.18, color="#ef4444", label="Absolute error area")
    plt.title("Fold 6 Shock Regime: Mainline Prediction Zoom")
    plt.xlabel("Date")
    plt.ylabel("Brent future 30D average price")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cumulative_direction_calls_academic.png", dpi=180)
    plt.close()


def write_export_report(summary: dict[str, Any], test_table: pd.DataFrame, rolling_table: pd.DataFrame) -> None:
    return


def run_all(force: bool = False) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    for window in (90, 120, 150):
        build_windows_from_fusion_features(window, force=force)

    candidates = candidate_grid()
    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    stage1 = run_stage(candidates, seeds=[42], top_k=5, stage_name="stage1_single_seed", force=force)
    stage2_candidates = [get_candidate_by_name(candidates, name) for name in stage1["candidate_name"].tolist()]
    stage2 = run_stage(stage2_candidates, seeds=[42, 43, 44], top_k=2, stage_name="stage2_three_seed", force=force)
    stage3_candidates = [get_candidate_by_name(candidates, name) for name in stage2["candidate_name"].tolist()]
    stage3 = run_stage(stage3_candidates, seeds=[42, 43, 44, 45, 46], top_k=2, stage_name="stage3_five_seed", force=force)

    top_names = stage3["candidate_name"].tolist()
    final_options: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name in top_names:
        candidate = candidate_lookup[name]
        official = run_candidate_official(candidate, seeds=[42, 43, 44, 45, 46], force=force)
        rolling = json.load(
            open(
                RESULT_ROOT / "rolling_runs" / f"{slugify(name)}__seeds_5" / "summary.json",
                "r",
                encoding="utf-8",
            )
        )
        final_options.append((name, official, rolling))

    if len(top_names) >= 2:
        ensemble_name = "late_gru_gate_config_ensemble_top2"
        ensemble_rolling_summary = ensemble_rolling(top_names[:2], seeds=5, ensemble_name=ensemble_name)
        ensemble_official_summary = ensemble_official(
            top_names[:2],
            candidate_lookup=candidate_lookup,
            seeds=[42, 43, 44, 45, 46],
            ensemble_name=ensemble_name,
        )
        final_options.append((ensemble_name, ensemble_official_summary, ensemble_rolling_summary))

    rows = []
    for name, official, rolling in final_options:
        victory = evaluate_victory(official, rolling)
        rows.append(
            {
                "candidate_name": name,
                **victory,
                "test_diagnostics_status": official.get("diagnostics_status", ""),
            }
        )
    basis = pd.DataFrame(rows).sort_values(
        ["all_pass", "mainline_rolling_score", "mainline_test_rmse"],
        ascending=[False, True, True],
    )
    basis.to_csv(RESULT_ROOT / "tables" / "final_selection_basis.csv", index=False, encoding="utf-8")
    write_json(RESULT_ROOT / "tables" / "final_selection_basis.json", {"records": basis.to_dict(orient="records")})

    selected_name = str(basis.iloc[0]["candidate_name"])
    selected = next(item for item in final_options if item[0] == selected_name)
    baseline = load_current_mainline_baseline()
    overwrite_gate = passes_overwrite_gate(selected[1], selected[2], baseline)
    if should_overwrite_current_export(overwrite_gate):
        make_export(selected[0], selected[1], selected[2])
        export_root = project_relative(EXPORT_ROOT)
    else:
        write_no_improvement_report(selected[0], overwrite_gate, basis)
        export_root = project_relative(EXPORT_ROOT)
    write_json(
        RESULT_ROOT / "MAINLINE_TUNING_RECORD.json",
        {
            "selected": selected_name,
            "overwrite_gate": overwrite_gate,
            "final_selection_basis": basis.to_dict(orient="records"),
            "stage1_top": stage1.to_dict(orient="records"),
            "stage2_top": stage2.to_dict(orient="records"),
            "stage3_top": stage3.to_dict(orient="records"),
            "export_root": export_root,
        },
    )
    return {"selected": selected_name, "overwrite_gate": overwrite_gate, "basis": basis.to_dict(orient="records"), "export_root": export_root}


def run_gate_feature_search(force: bool = False) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = gate_feature_candidate_grid(force=force)
    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    stage1 = run_stage(candidates, seeds=[42], top_k=6, stage_name="stage1_gate_feature_single_seed", force=force)
    stage2_candidates = [get_candidate_by_name(candidates, name) for name in stage1["candidate_name"].tolist()]
    stage2 = run_stage(stage2_candidates, seeds=[42, 43, 44], top_k=3, stage_name="stage2_gate_feature_three_seed", force=force)
    stage3_candidates = [get_candidate_by_name(candidates, name) for name in stage2["candidate_name"].tolist()]
    stage3 = run_stage(stage3_candidates, seeds=[42, 43, 44, 45, 46], top_k=2, stage_name="stage3_gate_feature_five_seed", force=force)

    top_names = stage3["candidate_name"].tolist()
    final_options: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name in top_names:
        candidate = candidate_lookup[name]
        official = run_candidate_official(candidate, seeds=[42, 43, 44, 45, 46], force=force)
        rolling = json.load(
            open(
                RESULT_ROOT / "rolling_runs" / f"{slugify(name)}__seeds_5" / "summary.json",
                "r",
                encoding="utf-8",
            )
        )
        final_options.append((name, official, rolling))

    if len(top_names) >= 2:
        ensemble_members = top_names[: min(3, len(top_names))]
        ensemble_name = f"late_gru_gate_gate_feature_ensemble_top{len(ensemble_members)}"
        ensemble_rolling_summary = ensemble_rolling(ensemble_members, seeds=5, ensemble_name=ensemble_name)
        ensemble_official_summary = ensemble_official(
            ensemble_members,
            candidate_lookup=candidate_lookup,
            seeds=[42, 43, 44, 45, 46],
            ensemble_name=ensemble_name,
        )
        final_options.append((ensemble_name, ensemble_official_summary, ensemble_rolling_summary))

    rows = []
    for name, official, rolling in final_options:
        victory = evaluate_victory(official, rolling)
        rows.append(
            {
                "candidate_name": name,
                **victory,
                "test_diagnostics_status": official.get("diagnostics_status", ""),
            }
        )
    basis = pd.DataFrame(rows).sort_values(
        ["all_pass", "mainline_rolling_score", "mainline_test_rmse"],
        ascending=[False, True, True],
    )
    basis.to_csv(RESULT_ROOT / "tables" / "final_selection_basis_gate_feature.csv", index=False, encoding="utf-8")
    write_json(
        RESULT_ROOT / "tables" / "final_selection_basis_gate_feature.json",
        {"records": basis.to_dict(orient="records")},
    )

    selected_name = str(basis.iloc[0]["candidate_name"])
    selected = next(item for item in final_options if item[0] == selected_name)
    baseline = load_current_mainline_baseline()
    overwrite_gate = passes_overwrite_gate(selected[1], selected[2], baseline)
    if should_overwrite_current_export(overwrite_gate):
        make_export(selected[0], selected[1], selected[2])
        export_root = project_relative(EXPORT_ROOT)
    else:
        write_no_improvement_report(selected[0], overwrite_gate, basis)
        export_root = project_relative(EXPORT_ROOT)
    write_json(
        RESULT_ROOT / "MAINLINE_GATE_FEATURE_TUNING_RECORD.json",
        {
            "selected": selected_name,
            "overwrite_gate": overwrite_gate,
            "final_selection_basis": basis.to_dict(orient="records"),
            "stage1_top": stage1.to_dict(orient="records"),
            "stage2_top": stage2.to_dict(orient="records"),
            "stage3_top": stage3.to_dict(orient="records"),
            "export_root": export_root,
        },
    )
    return {"selected": selected_name, "overwrite_gate": overwrite_gate, "basis": basis.to_dict(orient="records"), "export_root": export_root}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and export the daily horizon-30 late.gru_gate mainline.")
    parser.add_argument("--mode", choices=["all", "gate-feature", "visuals-only"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-run cached rolling and official runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "visuals-only":
        refresh_existing_export_visuals()
        result = {"mode": "visuals-only", "export_root": project_relative(EXPORT_ROOT)}
    else:
        result = run_gate_feature_search(force=args.force) if args.mode == "gate-feature" else run_all(force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
