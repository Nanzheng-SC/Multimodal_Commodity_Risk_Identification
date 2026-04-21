from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEMIXER_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = TIMEMIXER_ROOT.parent
import sys
for path in (TIMEMIXER_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.paths import OFFICIAL_RESULTS_ROOT, TIMEMIXER_MODALITY_COMPARISON_DIR, TIMEMIXER_MODEL_NAMES
from common.reporting import clean_directory, configure_matplotlib, save_json


PROJECT_ROOT = MODELING_ROOT.parent


def to_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _variant_label(input_variant: str) -> str:
    labels = {
        "fusion": "Multimodal",
        "text": "Text Only",
        "image": "Image Only",
        "structured": "Structured Only",
    }
    return labels[input_variant]


def _resolve_variant_dir(input_variant: str) -> Path:
    model_dir = OFFICIAL_RESULTS_ROOT / TIMEMIXER_MODEL_NAMES[input_variant]
    if not model_dir.exists():
        raise FileNotFoundError(f"Official directory not found for variant={input_variant}: {model_dir}")

    candidates = []
    for window_dir in sorted(model_dir.glob("window_*")):
        metrics_path = window_dir / "official_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        candidates.append((payload, window_dir))

    if not candidates:
        raise FileNotFoundError(f"No official_metrics.json found for variant={input_variant}: {model_dir}")

    selected_payload, selected_dir = min(
        candidates,
        key=lambda item: (
            0 if item[0].get("diagnostics_status", "FAIL") == "PASS" else 1,
            item[0]["deployed_run"]["final_valid_rmse"],
            item[0]["deployed_run"]["test_rmse"],
            item[0]["aggregate_metrics"]["test"].get("rmse_std", 0.0),
            -item[0]["aggregate_metrics"]["test"].get("direction_acc_mean", 0.0),
            item[0]["window_length"],
        ),
    )
    return selected_dir


def _load_variant_payload(input_variant: str) -> dict:
    official_dir = _resolve_variant_dir(input_variant)
    with open(official_dir / "official_metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    seed_df = pd.read_csv(official_dir / "rolling_seed_summary.csv")
    pred_df = pd.read_csv(official_dir / "best_run" / "predictions_test.csv")
    return {
        "input_variant": input_variant,
        "label": _variant_label(input_variant),
        "official_dir": official_dir,
        "metrics": metrics,
        "seed_df": seed_df,
        "pred_df": pred_df,
    }


def build_leaderboard(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record["metrics"]
        aggregate = metrics["aggregate_metrics"]
        deployed = metrics["deployed_run"]
        rows.append(
            {
                "input_variant": record["input_variant"],
                "label": record["label"],
                "window_length": metrics["window_length"],
                "input_dim": metrics["input_dim"],
                "diagnostics_status": metrics["diagnostics_status"],
                "deployed_final_valid_rmse": deployed["final_valid_rmse"],
                "deployed_test_rmse": deployed["test_rmse"],
                "deployed_test_mae": deployed["test_mae"],
                "test_rmse_mean": aggregate["test"]["rmse_mean"],
                "test_rmse_std": aggregate["test"]["rmse_std"],
                "test_mae_mean": aggregate["test"]["mae_mean"],
                "direction_acc_mean": aggregate["test"]["direction_acc_mean"],
                "official_dir": to_project_relative(record["official_dir"]),
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["deployed_final_valid_rmse", "deployed_test_rmse", "test_rmse_std", "direction_acc_mean", "input_variant"],
        ascending=[True, True, True, False, True],
    )
    return leaderboard.reset_index(drop=True)


def save_rmse_mae_plot(leaderboard: pd.DataFrame, output_dir: Path) -> None:
    configure_matplotlib()
    df = leaderboard.copy()
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, df["deployed_test_rmse"], width=width, label="Deployed Test RMSE")
    plt.bar(x + width / 2, df["deployed_test_mae"], width=width, label="Deployed Test MAE")
    plt.xticks(x, df["label"])
    plt.ylabel("Metric Value")
    plt.title("TimeMixer Modalities: Test RMSE / MAE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timemixer_modalities_rmse_mae.png", dpi=200)
    plt.close()


def save_stability_plot(leaderboard: pd.DataFrame, output_dir: Path) -> None:
    configure_matplotlib()
    df = leaderboard.copy()
    x = np.arange(len(df))
    plt.figure(figsize=(11, 5))
    plt.errorbar(x, df["test_rmse_mean"], yerr=df["test_rmse_std"], fmt="o", capsize=5, linewidth=2)
    plt.xticks(x, df["label"])
    plt.ylabel("Test RMSE mean ± std")
    plt.title("TimeMixer Modalities Stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "timemixer_modalities_stability.png", dpi=200)
    plt.close()


def save_prediction_overlay(records: list[dict], output_dir: Path) -> pd.DataFrame:
    configure_matplotlib()
    merged = None
    for record in records:
        pred_df = record["pred_df"].copy()
        pred_df = pred_df.rename(columns={"y_pred": f"pred_{record['input_variant']}"})
        keep_cols = ["month", "y_true", f"pred_{record['input_variant']}"]
        if merged is None:
            merged = pred_df[keep_cols]
        else:
            merged = merged.merge(pred_df[["month", f"pred_{record['input_variant']}"]], on="month", how="inner")

    if merged is None or merged.empty:
        raise ValueError("No shared test months found across modality predictions.")

    plot_month = pd.to_datetime(merged["month"])
    plt.figure(figsize=(13, 6))
    plt.plot(plot_month, merged["y_true"], marker="o", linewidth=2.8, color="black", label="Actual")
    for record in records:
        column = f"pred_{record['input_variant']}"
        plt.plot(plot_month, merged[column], marker="o", linewidth=1.8, label=record["label"])
    plt.title("TimeMixer Modalities Prediction Overlay")
    plt.xlabel("Month")
    plt.ylabel("Next Risk Label")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "timemixer_modalities_prediction_overlay.png", dpi=200)
    plt.close()
    return merged


def save_error_delta_plot(merged: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    configure_matplotlib()
    fusion_abs_error = np.abs(merged["pred_fusion"] - merged["y_true"])
    delta_df = pd.DataFrame({"month": merged["month"].astype(str)})
    for variant in ("text", "image", "structured"):
        abs_error = np.abs(merged[f"pred_{variant}"] - merged["y_true"])
        delta_df[f"delta_{variant}"] = abs_error - fusion_abs_error

    plot_month = pd.to_datetime(delta_df["month"])
    plt.figure(figsize=(13, 6))
    for variant in ("text", "image", "structured"):
        plt.plot(plot_month, delta_df[f"delta_{variant}"], marker="o", linewidth=1.8, label=f"{_variant_label(variant)} - Multimodal")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Absolute Error Delta vs Multimodal TimeMixer")
    plt.xlabel("Month")
    plt.ylabel("Abs Error Delta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "timemixer_modalities_error_delta_vs_fusion.png", dpi=200)
    plt.close()
    return delta_df


def main() -> None:
    clean_directory(TIMEMIXER_MODALITY_COMPARISON_DIR)
    records = [_load_variant_payload(variant) for variant in ("fusion", "text", "image", "structured")]
    leaderboard = build_leaderboard(records)
    leaderboard.to_csv(TIMEMIXER_MODALITY_COMPARISON_DIR / "leaderboard.csv", index=False, encoding="utf-8")
    save_json(
        TIMEMIXER_MODALITY_COMPARISON_DIR / "leaderboard.json",
        {"records": leaderboard.to_dict(orient="records")},
    )
    save_rmse_mae_plot(leaderboard, TIMEMIXER_MODALITY_COMPARISON_DIR)
    save_stability_plot(leaderboard, TIMEMIXER_MODALITY_COMPARISON_DIR)
    merged = save_prediction_overlay(records, TIMEMIXER_MODALITY_COMPARISON_DIR)
    merged.to_csv(TIMEMIXER_MODALITY_COMPARISON_DIR / "prediction_overlay_table.csv", index=False, encoding="utf-8")
    delta_df = save_error_delta_plot(merged, TIMEMIXER_MODALITY_COMPARISON_DIR)
    delta_df.to_csv(TIMEMIXER_MODALITY_COMPARISON_DIR / "error_delta_vs_fusion.csv", index=False, encoding="utf-8")
    print(f"Saved modality comparison to: {TIMEMIXER_MODALITY_COMPARISON_DIR.resolve()}")


if __name__ == "__main__":
    main()
