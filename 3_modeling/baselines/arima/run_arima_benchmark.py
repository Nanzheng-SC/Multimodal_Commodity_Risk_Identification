from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# statsmodels 0.14.x can import pandas' deprecate_kwarg with the pre-3.0
# calling convention. Keep this compatibility shim local to the ARIMA runner.
import pandas.util._decorators as _pd_decorators

_original_deprecate_kwarg = _pd_decorators.deprecate_kwarg


def _compat_deprecate_kwarg(*args, **kwargs):
    if args and isinstance(args[0], str):
        return _original_deprecate_kwarg(FutureWarning, *args, **kwargs)
    return _original_deprecate_kwarg(*args, **kwargs)


_pd_decorators.deprecate_kwarg = _compat_deprecate_kwarg

from statsmodels.tsa.arima.model import ARIMA


ARIMA_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = ARIMA_ROOT.parents[1]
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT, ARIMA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import build_diagnostics, compute_zero_baseline, metric_dict  # noqa: E402
from common.reporting import save_json  # noqa: E402
from arima_utils import (  # noqa: E402
    ARIMA_OFFICIAL_ROOT,
    DEFAULT_WINDOW_LENGTH,
    FREQUENCY,
    HORIZON_DAYS,
    LABEL_AVAILABILITY_RULE,
    MODEL_DISPLAY_NAME,
    MODEL_OFFICIAL_NAME,
    OFFICIAL_ROOT,
    build_common_folds,
    combine_split_frames,
    ensure_output_dir,
    fold_metrics_from_predictions,
    frame_metric_dict,
    label_available_history,
    load_mainline_split_frames,
    mirror_window_outputs,
    prediction_frame,
    project_relative,
    save_test_prediction_files,
    subset_period,
    summarize_fold_metrics,
    update_export_tables,
)


DEFAULT_ORDER_GRID = tuple((p, d, q) for p in range(4) for d in range(2) for q in range(4))


def _clean_series(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    return series[np.isfinite(series)]


def _fallback_forecast(train_values: np.ndarray, steps: int, mode: str = "mean") -> np.ndarray:
    clean = _clean_series(train_values)
    if len(clean) == 0:
        value = 0.0
    elif mode == "last":
        value = float(clean[-1])
    else:
        value = float(np.mean(clean))
    return np.full(int(steps), value, dtype=np.float32)


def forecast_residuals(train_values: np.ndarray, order: tuple[int, int, int], steps: int) -> np.ndarray:
    clean = _clean_series(train_values)
    if len(clean) < 10:
        return _fallback_forecast(clean, steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(
                clean,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            forecast = fitted.forecast(steps=int(steps))
        forecast = np.asarray(forecast, dtype=np.float32)
        if len(forecast) != steps or not np.isfinite(forecast).all():
            return _fallback_forecast(clean, steps)
        return forecast
    except Exception:
        return _fallback_forecast(clean, steps)


def select_order(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    order_grid: tuple[tuple[int, int, int], ...] = DEFAULT_ORDER_GRID,
    label: str = "fixed_valid",
) -> tuple[tuple[int, int, int], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_order: tuple[int, int, int] | None = None
    best_key: tuple[float, int, int, int] | None = None
    if train.empty:
        raise ValueError(f"No ARIMA training history available for {label}.")
    train_values = train["target_residual"].to_numpy(np.float32)
    for order in order_grid:
        pred_residual = forecast_residuals(train_values, order, len(valid))
        pred = prediction_frame(MODEL_DISPLAY_NAME, valid, pred_residual)
        metrics = frame_metric_dict(pred)
        row = {
            "selection_context": label,
            "p": int(order[0]),
            "d": int(order[1]),
            "q": int(order[2]),
            **metrics,
        }
        rows.append(row)
        key = (float(metrics["rmse"]), int(order[0]), int(order[1]), int(order[2]))
        if best_key is None or key < best_key:
            best_key = key
            best_order = order

    if best_order is None:
        best_order = (0, 0, 0)
        rows.append(
            {
                "selection_context": label,
                "p": 0,
                "d": 0,
                "q": 0,
                "mse": np.nan,
                "rmse": np.inf,
                "mae": np.nan,
                "mape": np.nan,
                "direction_acc": np.nan,
                "fallback": True,
            }
        )
    order_df = pd.DataFrame(rows).sort_values(["rmse", "p", "d", "q"], na_position="last").reset_index(drop=True)
    order_df["selected"] = (
        (order_df["p"] == best_order[0])
        & (order_df["d"] == best_order[1])
        & (order_df["q"] == best_order[2])
    )
    return best_order, order_df


def compute_zero_metrics(frame: pd.DataFrame) -> dict[str, float]:
    bundle = {
        "labels": frame["target_price"].to_numpy(np.float32),
        "reference": frame["reference_brent"].to_numpy(np.float32),
    }
    return compute_zero_baseline(bundle)


def run_fixed_test(splits: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = splits["train"].frame
    valid = splits["valid"].frame
    test = splits["test"].frame
    combined_train_valid = pd.concat([train, valid], ignore_index=True)
    valid_origin = str(valid["date"].iloc[0])
    test_origin = str(test["date"].iloc[0])
    order_history = label_available_history(combined_train_valid, valid_origin)
    selected_order, order_df = select_order(order_history, valid, label="fixed_valid")

    test_history = label_available_history(combined_train_valid, test_origin)
    test_residual_pred = forecast_residuals(test_history["target_residual"].to_numpy(np.float32), selected_order, len(test))
    test_pred = prediction_frame(MODEL_DISPLAY_NAME, test, test_residual_pred)
    fixed_metrics = frame_metric_dict(test_pred)

    valid_residual_pred = forecast_residuals(order_history["target_residual"].to_numpy(np.float32), selected_order, len(valid))
    valid_pred = prediction_frame(MODEL_DISPLAY_NAME, valid, valid_residual_pred)
    valid_metrics = frame_metric_dict(valid_pred)
    order_df["history_rows"] = int(len(order_history))
    order_df["forecast_origin"] = valid_origin
    order_df["label_availability_rule"] = LABEL_AVAILABILITY_RULE

    save_test_prediction_files(output_dir, test_pred)
    test_pred.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
    order_df.to_csv(output_dir / "order_selection.csv", index=False, encoding="utf-8")

    metrics = {
        "selected_order": list(selected_order),
        "label_availability_rule": LABEL_AVAILABILITY_RULE,
        "history_rows": {
            "validation_order_selection": int(len(order_history)),
            "test_forecast": int(len(test_history)),
        },
        "validation": valid_metrics,
        "test": fixed_metrics,
        "zero_baseline": {
            "final_valid": compute_zero_metrics(valid),
            "final_test": compute_zero_metrics(test),
        },
    }
    return metrics, test_pred, order_df


def run_rolling(combined: pd.DataFrame, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = build_common_folds(combined)
    predictions = []
    order_rows = []
    for _, fold_row in folds.iterrows():
        train = subset_period(combined, end=fold_row["train_end_exclusive"])
        valid = subset_period(combined, start=fold_row["valid_start"], end=fold_row["valid_end_exclusive"])
        eval_frame = subset_period(combined, start=fold_row["eval_start"], end=fold_row["eval_end_exclusive"])
        valid_history = label_available_history(train, str(fold_row["valid_start"]))
        selected_order, order_df = select_order(valid_history, valid, label=f"fold_{int(fold_row['fold'])}")
        order_df["fold"] = int(fold_row["fold"])
        order_df["history_rows"] = int(len(valid_history))
        order_df["forecast_origin"] = str(fold_row["valid_start"])
        order_df["label_availability_rule"] = LABEL_AVAILABILITY_RULE
        order_rows.append(order_df)

        train_valid = pd.concat([train, valid], ignore_index=True)
        eval_history = label_available_history(train_valid, str(fold_row["eval_start"]))
        pred_residual = forecast_residuals(
            eval_history["target_residual"].to_numpy(np.float32),
            selected_order,
            len(eval_frame),
        )
        fold_pred = prediction_frame(MODEL_DISPLAY_NAME, eval_frame, pred_residual, fold=int(fold_row["fold"]))
        fold_pred["selected_order"] = f"({selected_order[0]},{selected_order[1]},{selected_order[2]})"
        predictions.append(fold_pred)

    pred_df = pd.concat(predictions, ignore_index=True)
    fold_metrics = fold_metrics_from_predictions(pred_df)
    summary = summarize_fold_metrics(fold_metrics)
    folds.to_csv(output_dir / "split_manifest.csv", index=False, encoding="utf-8")
    pred_df.to_csv(output_dir / "rolling_predictions.csv", index=False, encoding="utf-8")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    order_selection = pd.concat(order_rows, ignore_index=True)
    # Keep the fixed-test order rows that were written first and append rolling rows later in main().
    return summary, pred_df, fold_metrics, order_selection


def build_official_metrics(
    window_length: int,
    fixed: dict[str, Any],
    rolling_summary: dict[str, Any],
    diagnostics: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "model": MODEL_OFFICIAL_NAME,
        "display_name": MODEL_DISPLAY_NAME,
        "input_variant": "target_residual_history",
        "frequency": FREQUENCY,
        "target": f"target_brent_avg_next_{HORIZON_DAYS}d",
        "target_definition": "target_brent_avg_next_30d - reference_brent",
        "target_mode": "residual",
        "forecast_horizon_days": HORIZON_DAYS,
        "window_length": int(window_length),
        "label_availability_rule": LABEL_AVAILABILITY_RULE,
        "time_series_root": project_relative(
            PROJECT_ROOT
            / "2_encoding_feature"
            / "outputs"
            / "daily"
            / "time_series_horizon30_mainline"
            / "late_gru_gate"
            / f"window_{window_length}"
        ),
        "split_strategy": "mainline_window90_fixed_test_plus_6fold_rolling_strict_label_availability",
        "selection_metric": "validation_rmse",
        "selected_order": fixed["selected_order"],
        "order_grid": [list(order) for order in DEFAULT_ORDER_GRID],
        "aggregate_metrics": {
            "final_valid": fixed["validation"],
            "test": fixed["test"],
            "rolling_validation": {
                key: value
                for key, value in rolling_summary.items()
                if key not in {"model", "selection_rule"}
            },
        },
        "deployed_run": {
            "mode": "single_arima_order_selected_by_validation_strict_label_availability",
            "final_valid_rmse": fixed["validation"]["rmse"],
            "final_valid_mae": fixed["validation"]["mae"],
            "final_valid_mape": fixed["validation"]["mape"],
            "direction_acc": fixed["test"]["direction_acc"],
            "test_rmse": fixed["test"]["rmse"],
            "test_mae": fixed["test"]["mae"],
            "test_mape": fixed["test"]["mape"],
        },
        "rolling_metrics": rolling_summary,
        "baseline_zero": fixed["zero_baseline"],
        "diagnostics_status": diagnostics["status"],
        "diagnostics": diagnostics,
        "elapsed_sec": float(elapsed_sec),
    }


def run_arima_benchmark(window_length: int = DEFAULT_WINDOW_LENGTH, update_export: bool = False) -> Path:
    import time

    start = time.time()
    output_dir = ARIMA_OFFICIAL_ROOT / f"window_{window_length}"
    ensure_output_dir(output_dir)

    splits = load_mainline_split_frames(window_length)
    combined = combine_split_frames(splits)

    fixed, _, fixed_order = run_fixed_test(splits, output_dir)
    rolling_summary, _, _, rolling_order = run_rolling(combined, output_dir)

    all_orders = pd.concat([fixed_order, rolling_order], ignore_index=True)
    all_orders.to_csv(output_dir / "order_selection.csv", index=False, encoding="utf-8")
    save_json(output_dir / "rolling_metrics.json", rolling_summary)

    diagnostics = build_diagnostics(
        y_pred=pd.read_csv(output_dir / "test_predictions.csv")["predicted_price"].to_numpy(np.float32),
        model_val_rmse=fixed["validation"]["rmse"],
        model_test_rmse=fixed["test"]["rmse"],
        zero_baseline_valid_rmse=fixed["zero_baseline"]["final_valid"]["rmse"],
        zero_baseline_test_rmse=fixed["zero_baseline"]["final_test"]["rmse"],
        aggregate_val_rmse_mean=fixed["validation"]["rmse"],
        aggregate_test_rmse_mean=fixed["test"]["rmse"],
    )
    official_metrics = build_official_metrics(
        window_length=window_length,
        fixed=fixed,
        rolling_summary=rolling_summary,
        diagnostics=diagnostics,
        elapsed_sec=time.time() - start,
    )
    save_json(output_dir / "official_metrics.json", official_metrics)
    pd.DataFrame(
        [
            {
                "model": MODEL_OFFICIAL_NAME,
                "display_name": MODEL_DISPLAY_NAME,
                "frequency": FREQUENCY,
                "window_length": window_length,
                "selected_order": str(tuple(fixed["selected_order"])),
                "final_valid_rmse": fixed["validation"]["rmse"],
                "test_rmse": fixed["test"]["rmse"],
                "test_mae": fixed["test"]["mae"],
                "test_mape": fixed["test"]["mape"],
                "test_direction_acc": fixed["test"]["direction_acc"],
                "rolling_score": rolling_summary["rolling_score"],
                "rolling_rmse_mean": rolling_summary["rmse_mean"],
                "rolling_direction_acc_mean": rolling_summary["direction_acc_mean"],
                "diagnostics_status": diagnostics["status"],
            }
        ]
    ).to_csv(output_dir / "official_metrics.csv", index=False, encoding="utf-8")

    mirror_window_outputs(output_dir, ARIMA_OFFICIAL_ROOT)
    if update_export:
        update_export_tables(output_dir, fixed["test"], rolling_summary)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official ARIMA residual baseline for daily horizon-30.")
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--update-export", action="store_true", help="Append ARIMA to the existing export leaderboards.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_arima_benchmark(window_length=args.window_length, update_export=args.update_export)
    print(f"Official ARIMA output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
