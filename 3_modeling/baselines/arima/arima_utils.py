from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARIMA_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = ARIMA_ROOT.parents[1]
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import metric_dict  # noqa: E402
from common.reporting import configure_matplotlib, save_json, save_prediction_artifacts  # noqa: E402
from project_shared.paths import STRUCTURED_DAILY_PATH  # noqa: E402
from project_shared.targets import REFERENCE_PRICE_COLUMN, compute_forward_average  # noqa: E402


FREQUENCY = "daily"
HORIZON_DAYS = 30
DEFAULT_WINDOW_LENGTH = 90
FOLD_COUNT = 6
FOLD_SIZE = 60
ROLLING_LAMBDA = 0.25
MODEL_DISPLAY_NAME = "ARIMA"
MODEL_OFFICIAL_NAME = "ARIMA-residual"
LABEL_AVAILABILITY_RULE = "target_end_strictly_before_forecast_origin"
MAINLINE_RUN_ID = "late_gru_gate_validation_selected_top2"
MODEL_DISPLAY_NAMES = {
    MAINLINE_RUN_ID: "TimeMixer (fusion; late.gru_gate)",
    "late.gru_concat": "TimeMixer (late.gru_concat)",
    "intermediate.gated": "TimeMixer (intermediate.gated)",
    "Structured": "TimeMixer (structured)",
    "Text": "TimeMixer (text)",
    "Image": "TimeMixer (image)",
    "Naive": "Naive",
    "HAR-no-leak": "HAR-no-leak",
    "LSTM": "LSTM",
    "ARIMA": "ARIMA",
}

WINDOW_ROOT = (
    PROJECT_ROOT
    / "2_encoding_feature"
    / "outputs"
    / "daily"
    / "time_series_horizon30_mainline"
    / "late_gru_gate"
)
OFFICIAL_ROOT = MODELING_ROOT / "results" / "official" / "daily_horizon30"
ARIMA_OFFICIAL_ROOT = OFFICIAL_ROOT / "arima_residual"
EXPORT_ROOT = MODELING_ROOT / "results" / "export" / "daily_horizon30_late_gru_gate_mainline_final"
STAT_ANALYSIS_ROOT = PROJECT_ROOT / "5_statistical_analysis"


@dataclass(frozen=True)
class SplitFrame:
    name: str
    frame: pd.DataFrame


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def display_model_name(run_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(run_id), str(run_id))


def normalize_export_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "run_id" not in result.columns and "model" in result.columns:
        result.insert(0, "run_id", result["model"].astype(str))
    if "model" in result.columns:
        source = result["run_id"] if "run_id" in result.columns else result["model"]
        result["model"] = source.astype(str).map(display_model_name)
    return result


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_structured_horizon_frame(horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    frame = pd.read_csv(STRUCTURED_DAILY_PATH)
    if "date" not in frame.columns:
        raise ValueError(f"{STRUCTURED_DAILY_PATH} must contain a date column.")
    if REFERENCE_PRICE_COLUMN not in frame.columns:
        raise ValueError(f"{STRUCTURED_DAILY_PATH} must contain {REFERENCE_PRICE_COLUMN}.")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    reference = frame["reference_brent"] if "reference_brent" in frame.columns else frame[REFERENCE_PRICE_COLUMN]
    frame["reference_brent"] = pd.to_numeric(reference, errors="coerce")
    frame[f"target_brent_avg_next_{horizon_days}d"] = compute_forward_average(
        pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce"),
        horizon=horizon_days,
    )
    frame["target_price"] = frame[f"target_brent_avg_next_{horizon_days}d"]
    frame["target_residual"] = frame["target_price"] - frame["reference_brent"]
    return frame


def load_mainline_indices(window_length: int = DEFAULT_WINDOW_LENGTH) -> dict[str, np.ndarray]:
    window_dir = WINDOW_ROOT / f"window_{window_length}"
    if not window_dir.exists():
        raise FileNotFoundError(f"Mainline window directory not found: {window_dir}")
    indices: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        path = window_dir / f"{split}_index.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing split index: {path}")
        indices[split] = np.load(path, allow_pickle=True).astype(str)
    return indices


def align_frame_to_index(frame: pd.DataFrame, index_values: np.ndarray) -> pd.DataFrame:
    lookup = frame.set_index("date", drop=False)
    missing = [value for value in index_values.astype(str) if value not in lookup.index]
    if missing:
        raise ValueError(f"Structured frame is missing {len(missing)} mainline dates; first={missing[0]}")
    aligned = lookup.loc[index_values.astype(str)].reset_index(drop=True).copy()
    required = ["target_price", "target_residual", "reference_brent"]
    aligned = aligned.dropna(subset=required).reset_index(drop=True)
    if len(aligned) != len(index_values):
        raise ValueError("Aligned frame lost rows after target/reference filtering.")
    return aligned


def load_mainline_split_frames(window_length: int = DEFAULT_WINDOW_LENGTH) -> dict[str, SplitFrame]:
    frame = load_structured_horizon_frame()
    indices = load_mainline_indices(window_length)
    return {
        split: SplitFrame(split, align_frame_to_index(frame, values))
        for split, values in indices.items()
    }


def combine_split_frames(splits: dict[str, SplitFrame]) -> pd.DataFrame:
    combined = pd.concat([splits[name].frame for name in ("train", "valid", "test")], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    return combined


def subset_period(frame: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates < pd.Timestamp(end)
    return frame.loc[mask].reset_index(drop=True).copy()


def label_available_history(
    frame: pd.DataFrame,
    origin_date: str,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Return residual labels observable before a forecast origin.

    The residual target for date t uses Brent through t + horizon_days. For a
    strict classical baseline, labels whose target window reaches the forecast
    origin are excluded from the ARIMA history.
    """
    dates = pd.to_datetime(frame["date"])
    origin = pd.Timestamp(origin_date)
    label_end = dates + pd.Timedelta(days=int(horizon_days))
    mask = label_end < origin
    return frame.loc[mask].reset_index(drop=True).copy()


def build_common_folds(combined: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(combined["date"])
    if len(dates) < FOLD_COUNT * FOLD_SIZE + FOLD_SIZE:
        raise ValueError("Not enough samples for the official 6-fold rolling plan.")
    eval_dates = dates.iloc[-FOLD_COUNT * FOLD_SIZE :].reset_index(drop=True)
    rows = []
    for fold_idx in range(FOLD_COUNT):
        fold_eval = eval_dates.iloc[fold_idx * FOLD_SIZE : (fold_idx + 1) * FOLD_SIZE]
        valid_end = fold_eval.iloc[0]
        valid_start = dates[dates < valid_end].iloc[-FOLD_SIZE]
        train_end = valid_start
        rows.append(
            {
                "fold": fold_idx + 1,
                "train_start": str(dates.iloc[0].date()),
                "train_end_exclusive": str(train_end.date()),
                "valid_start": str(valid_start.date()),
                "valid_end_exclusive": str(valid_end.date()),
                "eval_start": str(fold_eval.iloc[0].date()),
                "eval_end_inclusive": str(fold_eval.iloc[-1].date()),
                "eval_end_exclusive": str((fold_eval.iloc[-1] + pd.Timedelta(days=1)).date()),
                "eval_days": int(len(fold_eval)),
            }
        )
    return pd.DataFrame(rows)


def frame_metric_dict(frame: pd.DataFrame, pred_column: str = "predicted_price") -> dict[str, float]:
    return metric_dict(
        frame[pred_column].to_numpy(np.float32),
        frame["target_price"].to_numpy(np.float32),
        frame["reference_brent"].to_numpy(np.float32),
    )


def prediction_frame(model_name: str, source: pd.DataFrame, predicted_residual: np.ndarray, fold: int | None = None) -> pd.DataFrame:
    pred = source[["date", "reference_brent", "target_price", "target_residual"]].copy()
    pred["predicted_residual"] = np.asarray(predicted_residual, dtype=np.float32)
    pred["predicted_price"] = pred["reference_brent"].to_numpy(np.float32) + pred["predicted_residual"].to_numpy(np.float32)
    pred["error"] = pred["predicted_price"] - pred["target_price"]
    pred["abs_error"] = pred["error"].abs()
    pred["model"] = model_name
    if fold is not None:
        pred.insert(0, "fold", int(fold))
    ordered = [
        column
        for column in (
            "model",
            "fold",
            "date",
            "reference_brent",
            "target_price",
            "predicted_price",
            "target_residual",
            "predicted_residual",
            "error",
            "abs_error",
        )
        if column in pred.columns
    ]
    return pred[ordered]


def save_test_prediction_files(output_dir: Path, pred: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
    best_run = output_dir / "best_run"
    save_prediction_artifacts(
        best_run,
        pred["date"].to_numpy(str),
        pred["target_price"].to_numpy(np.float32),
        pred["predicted_price"].to_numpy(np.float32),
        "ARIMA residual baseline",
        frequency="daily",
    )

    # Add the richer residual columns next to the standard comparison file.
    rich = pred.rename(columns={"target_price": "y_true", "predicted_price": "y_pred"}).copy()
    rich[["date", "y_true", "y_pred", "error", "abs_error", "reference_brent", "target_residual", "predicted_residual"]].to_csv(
        best_run / "predictions_test_with_residuals.csv",
        index=False,
        encoding="utf-8",
    )


def fold_metrics_from_predictions(predictions: pd.DataFrame, model_name: str = MODEL_DISPLAY_NAME) -> pd.DataFrame:
    rows = []
    for fold, group in predictions.groupby("fold", sort=True):
        metrics = frame_metric_dict(group)
        rows.append({"model": model_name, "fold": int(fold), **metrics, "n": int(len(group))})
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics: pd.DataFrame, model_name: str = MODEL_DISPLAY_NAME) -> dict[str, Any]:
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
        "selection_rule": "single-variable ARIMA residual baseline with strict label-availability lag",
    }


def update_export_tables(arima_dir: Path, fixed_metrics: dict[str, float], rolling_summary: dict[str, Any]) -> None:
    table_dir = EXPORT_ROOT / "tables"
    if not table_dir.exists():
        raise FileNotFoundError(f"Export table directory not found: {table_dir}")

    test_path = table_dir / "final_test_leaderboard.csv"
    rolling_path = table_dir / "rolling_leaderboard.csv"
    basis_path = table_dir / "final_selection_basis.csv"

    test = normalize_export_model_columns(pd.read_csv(test_path))
    test = test[test["run_id"].astype(str) != MODEL_DISPLAY_NAME].copy()
    test = pd.concat(
        [
            test,
            pd.DataFrame(
                [
                    {
                        "run_id": MODEL_DISPLAY_NAME,
                        "model": MODEL_DISPLAY_NAME,
                        "test_rmse_mean": fixed_metrics["rmse"],
                        "test_mae_mean": fixed_metrics["mae"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("test_rmse_mean")
    test.to_csv(test_path, index=False, encoding="utf-8")

    rolling = normalize_export_model_columns(pd.read_csv(rolling_path))
    rolling = rolling[rolling["run_id"].astype(str) != MODEL_DISPLAY_NAME].copy()
    rolling_row = {column: rolling_summary.get(column, np.nan) for column in rolling.columns}
    rolling_row["run_id"] = MODEL_DISPLAY_NAME
    rolling_row["model"] = MODEL_DISPLAY_NAME
    rolling = pd.concat([rolling, pd.DataFrame([rolling_row])], ignore_index=True).sort_values("rolling_score")
    rolling.to_csv(rolling_path, index=False, encoding="utf-8")

    basis = normalize_export_model_columns(pd.read_csv(basis_path))
    basis = basis[basis["run_id"].astype(str) != MODEL_DISPLAY_NAME].copy()
    basis = pd.concat(
        [
            basis,
            pd.DataFrame(
                [
                    {
                        "run_id": MODEL_DISPLAY_NAME,
                        "model": MODEL_DISPLAY_NAME,
                        "role": "baseline_control",
                        "test_rmse_mean": fixed_metrics["rmse"],
                        "rolling_rmse_mean": rolling_summary["rmse_mean"],
                        "rolling_score": rolling_summary["rolling_score"],
                        "mainline_pass": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    # The paper's final selection basis keeps the official mainline first while
    # baseline controls remain sorted by the same rolling criterion.
    basis["role_order"] = np.where(basis["role"].astype(str) == "mainline", 0, 1)
    basis = basis.sort_values(["role_order", "rolling_score"], na_position="last").drop(columns=["role_order"])
    basis.to_csv(basis_path, index=False, encoding="utf-8")

    summary_path = EXPORT_ROOT / "EXPORT_SUMMARY.json"
    summary = load_json(summary_path)
    summary["arima_baseline"] = {
        "model": MODEL_OFFICIAL_NAME,
        "display_name": MODEL_DISPLAY_NAME,
        "official_dir": project_relative(arima_dir),
        "fixed_test": fixed_metrics,
        "rolling": {
            key: value
            for key, value in rolling_summary.items()
            if key not in {"selection_rule"}
        },
        "target": "target_brent_avg_next_30d - reference_brent",
        "label_availability_rule": LABEL_AVAILABILITY_RULE,
        "note": "Recorded as a baseline control under the same benchmark table semantics; final mainline selection is unchanged.",
    }
    summary["final_model_display_name"] = display_model_name(summary.get("final_name", MAINLINE_RUN_ID))
    summary["final_run_id"] = summary.get("final_name", MAINLINE_RUN_ID)
    summary.setdefault("export_files", {})["arima_official_dir"] = project_relative(arima_dir)
    summary.setdefault("export_files", {})["statistical_analysis_root"] = project_relative(STAT_ANALYSIS_ROOT)
    write_json(summary_path, summary)


def mirror_window_outputs(window_dir: Path, root_dir: Path) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "official_metrics.json",
        "official_metrics.csv",
        "test_predictions.csv",
        "rolling_metrics.json",
        "rolling_predictions.csv",
        "fold_metrics.csv",
        "split_manifest.csv",
        "order_selection.csv",
    ):
        copy_file_if_exists(window_dir / name, root_dir / name)


def save_simple_rolling_plots(rolling_summary: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    if rolling_summary.empty:
        return
    configure_matplotlib()
    frame = rolling_summary.sort_values("rolling_score", ascending=True).copy()
    plt.figure(figsize=(11, 6))
    colors = ["#276749" if str(model).startswith("TimeMixer") else "#d1dbe2" for model in frame["model"]]
    plt.barh(frame["model"], frame["rolling_score"], color=colors)
    plt.xlabel("Rolling score = RMSE mean + 0.25 * RMSE std")
    plt.title("Rolling Robustness Score")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "export_rolling_score_context.png", dpi=180)
    plt.close()
