from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODELING_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import aggregate_metric_dicts, build_diagnostics, compute_zero_baseline, metric_dict
from common.paths import official_model_dir, time_series_root
from common.reporting import clean_directory, copy_best_run_visuals, save_json, save_prediction_artifacts, save_seed_metric_plot
from common.window_data import (
    build_reference_lookup,
    build_rolling_plan,
    combine_splits,
    compute_future_window,
    load_split_arrays,
    with_reference,
)
from project_shared.frequency import normalize_frequency


SEED_LIST = [42, 43, 44]


def har_windows_for_frequency(frequency: str) -> tuple[int, int, int]:
    frequency = normalize_frequency(frequency)
    return (1, 7, 30) if frequency == "daily" else (1, 3, 12)


def build_har_design(labels: np.ndarray, history_windows: tuple[int, int, int], start_index: int, end_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_window = max(history_windows)
    features = []
    targets = []
    indices = []
    for idx in range(max_window, end_index):
        if idx < start_index:
            continue
        row = [float(np.mean(labels[idx - window : idx])) for window in history_windows]
        features.append(row)
        targets.append(labels[idx])
        indices.append(idx)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    weights = coef[:-1].astype(np.float32)
    bias = float(coef[-1])
    return weights, bias


def predict_linear_regression(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    pred = x @ weights + bias
    return np.maximum(pred.astype(np.float32), 0.0)


def evaluate_range(
    labels: np.ndarray,
    reference: np.ndarray,
    history_windows: tuple[int, int, int],
    start_index: int,
    end_index: int,
    weights: np.ndarray,
    bias: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    x_eval, y_eval, indices = build_har_design(labels, history_windows, start_index, end_index)
    pred = predict_linear_regression(x_eval, weights, bias)
    metrics = metric_dict(pred, y_eval, reference[indices])
    return metrics, pred, indices


def run_har_benchmark(
    window_length: int | None = None,
    results_dir: Path | None = None,
    input_variant: str = "fusion",
    frequency: str = "monthly",
) -> Path:
    frequency = normalize_frequency(frequency)
    history_windows = har_windows_for_frequency(frequency)
    required_window = max(history_windows)
    if window_length is not None and window_length != required_window:
        raise ValueError(f"HAR baseline requires window_length={required_window} for frequency={frequency}.")
    window_length = required_window

    official_dir = results_dir or official_model_dir("har", window_length, frequency)
    clean_directory(official_dir)

    resolved_root = time_series_root(input_variant, frequency)
    split_arrays = with_reference(
        load_split_arrays(window_length, root_path=resolved_root, frequency=frequency),
        build_reference_lookup(frequency),
        frequency=frequency,
    )
    combined = combine_splits(split_arrays)
    combined["reference"] = np.concatenate(
        [split_arrays["train"]["reference"], split_arrays["valid"]["reference"], split_arrays["test"]["reference"]],
        axis=0,
    )
    rolling_plan = build_rolling_plan(combined, frequency=frequency)
    final_plan = rolling_plan["final_plan"]

    labels = combined["labels"]
    index_values = combined["index"]
    reference = combined["reference"]

    seed_rows = []
    seed_metric_rolling = []
    seed_metric_final_valid = []
    seed_metric_test = []
    best_seed_payload = None

    final_train_end = len(final_plan["train"]["labels"])
    final_valid_end = final_train_end + len(final_plan["valid"]["labels"])
    final_test_end = final_valid_end + len(final_plan["test"]["labels"])

    for seed in SEED_LIST:
        fold_records = []
        for fold in rolling_plan["validation_folds"]:
            train_end = len(fold["train"]["labels"])
            valid_end = train_end + len(fold["valid"]["labels"])
            x_train, y_train, _ = build_har_design(labels, history_windows, 0, train_end)
            weights, bias = fit_linear_regression(x_train, y_train)
            valid_metrics, _, _ = evaluate_range(labels, reference, history_windows, train_end, valid_end, weights, bias)
            fold_records.append(valid_metrics)

        x_final_train, y_final_train, _ = build_har_design(labels, history_windows, 0, final_train_end)
        weights, bias = fit_linear_regression(x_final_train, y_final_train)

        train_metrics, _, _ = evaluate_range(labels, reference, history_windows, max(history_windows), final_train_end, weights, bias)
        final_valid_metrics, _, valid_indices = evaluate_range(labels, reference, history_windows, final_train_end, final_valid_end, weights, bias)
        test_metrics, test_pred, test_indices = evaluate_range(labels, reference, history_windows, final_valid_end, final_test_end, weights, bias)

        row = {
            "seed": seed,
            "rolling_val_rmse_mean": float(np.mean([record["rmse"] for record in fold_records])),
            "rolling_val_rmse_std": float(np.std([record["rmse"] for record in fold_records])),
            "rolling_val_mae_mean": float(np.mean([record["mae"] for record in fold_records])),
            "rolling_val_mae_std": float(np.std([record["mae"] for record in fold_records])),
            "rolling_val_mape_mean": float(np.mean([record["mape"] for record in fold_records])),
            "rolling_val_mse_mean": float(np.mean([record["mse"] for record in fold_records])),
            "rolling_val_direction_acc_mean": float(np.mean([record["direction_acc"] for record in fold_records])),
            "final_valid_rmse": float(final_valid_metrics["rmse"]),
            "final_valid_mae": float(final_valid_metrics["mae"]),
            "final_valid_mape": float(final_valid_metrics["mape"]),
            "final_valid_mse": float(final_valid_metrics["mse"]),
            "final_valid_direction_acc": float(final_valid_metrics["direction_acc"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_mape": float(test_metrics["mape"]),
            "test_mse": float(test_metrics["mse"]),
            "test_direction_acc": float(test_metrics["direction_acc"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_mape": float(train_metrics["mape"]),
            "train_mse": float(train_metrics["mse"]),
            "train_direction_acc": float(train_metrics["direction_acc"]),
            "best_epoch_final": 0,
        }
        seed_rows.append(row)
        seed_metric_rolling.append(
            {
                "rmse": row["rolling_val_rmse_mean"],
                "mae": row["rolling_val_mae_mean"],
                "mape": row["rolling_val_mape_mean"],
                "mse": row["rolling_val_mse_mean"],
                "direction_acc": row["rolling_val_direction_acc_mean"],
            }
        )
        seed_metric_final_valid.append(final_valid_metrics)
        seed_metric_test.append(test_metrics)

        candidate_payload = {
            "summary": row,
            "weights": weights.copy(),
            "bias": bias,
            "test_pred": test_pred.copy(),
            "test_indices": test_indices.copy(),
            "valid_indices": valid_indices.copy(),
        }
        if best_seed_payload is None or (
            row["final_valid_rmse"],
            row["test_rmse"],
            row["seed"],
        ) < (
            best_seed_payload["summary"]["final_valid_rmse"],
            best_seed_payload["summary"]["test_rmse"],
            best_seed_payload["summary"]["seed"],
        ):
            best_seed_payload = candidate_payload

    seed_df = pd.DataFrame(seed_rows).sort_values(["final_valid_rmse", "test_rmse", "seed"]).reset_index(drop=True)
    seed_df.to_csv(official_dir / "rolling_seed_summary.csv", index=False, encoding="utf-8")
    save_seed_metric_plot(official_dir, seed_df, f"HAR-{frequency}")

    best_run_dir = official_dir / "best_run"
    best_run_dir.mkdir(parents=True, exist_ok=True)
    test_indices = best_seed_payload["test_indices"]
    save_prediction_artifacts(
        best_run_dir,
        index_values[test_indices],
        labels[test_indices],
        best_seed_payload["test_pred"],
        "HAR",
        frequency=frequency,
    )
    copy_best_run_visuals(best_run_dir, official_dir)

    last_idx = len(labels)
    future_x = np.asarray(
        [[float(np.mean(labels[last_idx - window : last_idx])) for window in history_windows]],
        dtype=np.float32,
    )
    future_pred = float(predict_linear_regression(future_x, best_seed_payload["weights"], best_seed_payload["bias"])[0])
    forecast_window = compute_future_window(str(index_values[last_idx - 1]), frequency=frequency)
    future_payload = {
        "window_length": window_length,
        "input_variant": input_variant,
        "input_window_start": str(index_values[last_idx - window_length]),
        "input_window_end": str(index_values[last_idx - 1]),
        "predicted_value": future_pred,
        "model": "HAR",
        "fusion_method": "late.gru_gate" if input_variant == "fusion" else None,
        "frequency": frequency,
        **forecast_window,
    }
    save_json(best_run_dir / "future_forecast.json", future_payload)
    save_json(official_dir / "future_forecast.json", future_payload)

    zero_validation_metrics = [compute_zero_baseline(fold["valid"]) for fold in rolling_plan["validation_folds"]]
    zero_baseline = {
        "rolling_validation": aggregate_metric_dicts(zero_validation_metrics),
        "final_valid": compute_zero_baseline(final_plan["valid"]),
        "final_test": compute_zero_baseline(final_plan["test"]),
    }
    save_json(official_dir / "baseline_zero.json", zero_baseline)

    aggregate_rolling = aggregate_metric_dicts(seed_metric_rolling)
    aggregate_final_valid = aggregate_metric_dicts(seed_metric_final_valid)
    aggregate_test = aggregate_metric_dicts(seed_metric_test)
    diagnostics = build_diagnostics(
        y_pred=best_seed_payload["test_pred"],
        model_val_rmse=best_seed_payload["summary"]["final_valid_rmse"],
        model_test_rmse=best_seed_payload["summary"]["test_rmse"],
        zero_baseline_valid_rmse=zero_baseline["final_valid"]["rmse"],
        zero_baseline_test_rmse=zero_baseline["final_test"]["rmse"],
        aggregate_val_rmse_mean=aggregate_final_valid["rmse_mean"],
        aggregate_test_rmse_mean=aggregate_test["rmse_mean"],
    )
    save_json(official_dir / "diagnostics.json", diagnostics)

    official_metrics = {
        "model": "HAR",
        "input_variant": input_variant,
        "window_length": window_length,
        "input_dim": len(history_windows),
        "frequency": frequency,
        "target": "target_brent_avg_next_7d",
        "split_strategy": "rolling_origin",
        "seed_list": SEED_LIST,
        "selection_metric": "final_valid_rmse_mean",
        "aggregate_metrics": {
            "rolling_validation": aggregate_rolling,
            "final_valid": aggregate_final_valid,
            "test": aggregate_test,
        },
        "deployed_run": {
            "seed": int(best_seed_payload["summary"]["seed"]),
            "rolling_val_rmse_mean": float(best_seed_payload["summary"]["rolling_val_rmse_mean"]),
            "final_valid_rmse": float(best_seed_payload["summary"]["final_valid_rmse"]),
            "final_valid_mae": float(best_seed_payload["summary"]["final_valid_mae"]),
            "final_valid_mape": float(best_seed_payload["summary"]["final_valid_mape"]),
            "test_rmse": float(best_seed_payload["summary"]["test_rmse"]),
            "test_mae": float(best_seed_payload["summary"]["test_mae"]),
            "test_mape": float(best_seed_payload["summary"]["test_mape"]),
            "best_epoch_final": 0,
        },
        "baseline_zero": zero_baseline,
        "diagnostics_status": diagnostics["status"],
        "config": {"history_windows": list(history_windows)},
    }
    save_json(official_dir / "official_metrics.json", official_metrics)
    pd.DataFrame(
        [
            {
                "model": "HAR",
                "frequency": frequency,
                "window_length": window_length,
                "selection_metric": "final_valid_rmse_mean",
                "final_valid_rmse_mean": aggregate_final_valid["rmse_mean"],
                "final_valid_rmse_std": aggregate_final_valid["rmse_std"],
                "test_rmse_mean": aggregate_test["rmse_mean"],
                "test_rmse_std": aggregate_test["rmse_std"],
                "deployed_final_valid_rmse": best_seed_payload["summary"]["final_valid_rmse"],
                "deployed_test_rmse": best_seed_payload["summary"]["test_rmse"],
                "diagnostics_status": diagnostics["status"],
            }
        ]
    ).to_csv(official_dir / "official_metrics.csv", index=False, encoding="utf-8")
    return official_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official dual-frequency HAR benchmark.")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--input-variant", type=str, default="fusion")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    output_dir = run_har_benchmark(
        window_length=args.window_length,
        results_dir=results_dir,
        input_variant=args.input_variant,
        frequency=args.frequency,
    )
    print(f"Official HAR output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
