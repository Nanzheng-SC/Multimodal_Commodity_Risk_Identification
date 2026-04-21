from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_shared.frequency import index_column_for_frequency, normalize_frequency


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def clean_directory(path: Path, keep_names: set[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    keep = keep_names or set()
    for item in path.iterdir():
        if item.name in keep:
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except PermissionError:
            continue


def save_prediction_artifacts(
    output_dir: Path,
    index_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str,
    frequency: str = "monthly",
) -> None:
    frequency = normalize_frequency(frequency)
    index_column = index_column_for_frequency(frequency)
    x_label = "Date" if frequency == "daily" else "Month"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    df = pd.DataFrame({index_column: index_values, "y_true": y_true, "y_pred": y_pred})
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = np.abs(df["error"])
    df.to_csv(output_dir / "predictions_test.csv", index=False, encoding="utf-8")
    df.to_csv(output_dir / "predictions_test_with_errors.csv", index=False, encoding="utf-8")

    plot_index = pd.to_datetime(df[index_column])

    plt.figure(figsize=(12, 6))
    plt.plot(plot_index, df["y_true"], marker="o", linewidth=2, label="Actual")
    plt.plot(plot_index, df["y_pred"], marker="o", linewidth=2, alpha=0.85, label="Predicted")
    plt.title(f"{title_prefix} Test Prediction Curve")
    plt.xlabel(x_label)
    plt.ylabel("Brent Next-7D Average Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["error"], bins=min(12, max(6, len(df) // 2)), alpha=0.8)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{title_prefix} Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(df["y_true"], df["y_pred"], alpha=0.8)
    axis_min = min(df["y_true"].min(), df["y_pred"].min())
    axis_max = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.5)
    plt.title(f"{title_prefix} Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_pred_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_index.astype(str), df["abs_error"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{title_prefix} Absolute Error by Period")
    plt.xlabel(x_label)
    plt.ylabel("|Prediction Error|")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "absolute_error_by_period.png", dpi=200)
    plt.close()


def copy_best_run_visuals(best_run_dir: Path, official_dir: Path) -> None:
    mapping = {
        "prediction_curve.png": "official_prediction_curve.png",
        "residual_histogram.png": "official_residual_histogram.png",
        "actual_vs_pred_scatter.png": "official_actual_vs_pred_scatter.png",
        "absolute_error_by_period.png": "official_absolute_error_by_period.png",
    }
    for src_name, dst_name in mapping.items():
        src_path = best_run_dir / src_name
        if src_path.exists():
            shutil.copy2(src_path, official_dir / dst_name)


def save_seed_metric_plot(output_dir: Path, seed_df: pd.DataFrame, model_label: str) -> None:
    configure_matplotlib()
    plt.figure(figsize=(10, 5))
    x = np.arange(len(seed_df))
    width = 0.35
    plt.bar(x - width / 2, seed_df["final_valid_rmse"], width=width, label="final valid RMSE")
    plt.bar(x + width / 2, seed_df["test_rmse"], width=width, label="test RMSE")
    plt.xticks(x, seed_df["seed"].astype(str))
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title(f"{model_label} Final Valid vs Test RMSE by Seed")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "seed_metric_comparison.png", dpi=200)
    plt.close()
