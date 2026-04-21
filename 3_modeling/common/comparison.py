from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.reporting import configure_matplotlib, save_json


def _prediction_index_column(df: pd.DataFrame) -> str:
    for candidate in ("date", "month"):
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        if column not in {"y_true", "y_pred", "error", "abs_error"}:
            return column
    raise ValueError("Prediction dataframe does not contain an index column.")


def load_official_record(model_name: str, official_dir: Path) -> dict:
    metrics_path = official_dir / "official_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    deployed = payload["deployed_run"]
    aggregate = payload["aggregate_metrics"]
    return {
        "model": model_name,
        "frequency": payload.get("frequency", "monthly"),
        "window_length": payload["window_length"],
        "selection_metric": payload["selection_metric"],
        "diagnostics_status": payload["diagnostics_status"],
        "diagnostics_rank": 0 if payload["diagnostics_status"] == "PASS" else 1,
        "final_valid_rmse_mean": aggregate["final_valid"]["rmse_mean"],
        "final_valid_rmse_std": aggregate["final_valid"]["rmse_std"],
        "test_rmse_mean": aggregate["test"]["rmse_mean"],
        "test_rmse_std": aggregate["test"]["rmse_std"],
        "test_mae_mean": aggregate["test"]["mae_mean"],
        "test_mae_std": aggregate["test"]["mae_std"],
        "test_mape_mean": aggregate["test"].get("mape_mean"),
        "test_mape_std": aggregate["test"].get("mape_std"),
        "direction_acc_mean": aggregate["test"]["direction_acc_mean"],
        "direction_acc_std": aggregate["test"]["direction_acc_std"],
        "deployed_final_valid_rmse": deployed["final_valid_rmse"],
        "deployed_test_rmse": deployed["test_rmse"],
        "deployed_test_mae": deployed["test_mae"],
        "deployed_test_mape": deployed.get("test_mape"),
        "best_seed": deployed.get("seed"),
    }


def save_leaderboard(records: list[dict], output_dir: Path) -> pd.DataFrame:
    leaderboard = pd.DataFrame(records).sort_values(
        ["diagnostics_rank", "deployed_test_rmse", "deployed_final_valid_rmse", "test_rmse_std", "direction_acc_mean", "model"],
        ascending=[True, True, True, True, False, True],
    )
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False, encoding="utf-8")
    save_json(output_dir / "leaderboard.json", {"records": leaderboard.to_dict(orient="records")})
    return leaderboard


def save_metric_barplots(leaderboard: pd.DataFrame, output_dir: Path) -> None:
    configure_matplotlib()
    df = leaderboard.reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["deployed_final_valid_rmse"], width=width, label="deployed final valid RMSE")
    plt.bar(x + width / 2, df["deployed_test_rmse"], width=width, label="deployed test RMSE")
    plt.xticks(x, df["model"])
    plt.ylabel("RMSE")
    plt.title("Model RMSE Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metric_barplots.png", dpi=200)
    plt.close()


def save_stability_plot(leaderboard: pd.DataFrame, output_dir: Path) -> None:
    configure_matplotlib()
    df = leaderboard.reset_index(drop=True)
    x = np.arange(len(df))
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x,
        df["test_rmse_mean"],
        yerr=df["test_rmse_std"],
        fmt="o",
        capsize=5,
        linewidth=2,
    )
    plt.xticks(x, df["model"])
    plt.ylabel("Test RMSE mean ± std")
    plt.title("Model Stability Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "stability_comparison.png", dpi=200)
    plt.close()


def save_prediction_overlay(model_dirs: dict[str, Path], output_dir: Path) -> None:
    configure_matplotlib()
    merged = None
    metric_summary = {}
    index_column = None
    for model_name, model_dir in model_dirs.items():
        pred_path = model_dir / "best_run" / "predictions_test.csv"
        df = pd.read_csv(pred_path)
        current_index_column = _prediction_index_column(df)
        if index_column is None:
            index_column = current_index_column
        errors = df["y_pred"] - df["y_true"]
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
        metric_summary[model_name] = {"rmse": rmse, "mae": mae}
        df = df.rename(columns={"y_pred": f"pred_{model_name.lower()}"})
        keep_cols = [index_column, "y_true", f"pred_{model_name.lower()}"]
        if merged is None:
            merged = df[keep_cols]
        else:
            merged = merged.merge(df[[index_column, f"pred_{model_name.lower()}"]], on=index_column, how="inner")

    if merged is None or merged.empty or index_column is None:
        return

    plot_index = pd.to_datetime(merged[index_column])
    x_label = "Date" if index_column == "date" else "Month"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(plot_index, merged["y_true"], marker="o", linewidth=2.8, color="black", label="Actual")
    for model_name in model_dirs:
        axes[0].plot(
            plot_index,
            merged[f"pred_{model_name.lower()}"],
            marker="o",
            linewidth=1.8,
            label=f"{model_name} (RMSE={metric_summary[model_name]['rmse']:.4f}, MAE={metric_summary[model_name]['mae']:.4f})",
        )
    axes[0].set_title("Test Prediction Comparison")
    axes[0].set_ylabel("Brent Next-7D Avg")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for model_name in model_dirs:
        abs_error = np.abs(merged[f"pred_{model_name.lower()}"] - merged["y_true"])
        axes[1].plot(plot_index, abs_error, marker="o", linewidth=1.8, label=f"{model_name} abs error")
    axes[1].set_title("Absolute Error by Period")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "test_prediction_overlay.png", dpi=200)
    plt.close()
