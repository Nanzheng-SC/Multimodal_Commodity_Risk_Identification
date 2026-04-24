from __future__ import annotations

from textwrap import shorten

import numpy as np
import pandas as pd

from analysis_common import (
    DATA_DIR,
    FIGURE_DIR,
    MAINLINE_RUN_ID,
    PALETTE,
    TABLE_DIR,
    configure_paper_style,
    display_model_name,
    ensure_dirs,
    load_chapter3_structured,
    model_color,
    save_figure,
    style_axis,
)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def _short_label(value: str, width: int = 42) -> str:
    return shorten(str(value), width=width, placeholder="...")


def plot_macro_standardized_lines() -> None:
    import matplotlib.pyplot as plt

    frame = load_chapter3_structured().copy()
    frame["date"] = pd.to_datetime(frame["date"])
    columns = [column for column in ["Brent", "WTI", "USD_Index", "EPU", "GPR"] if column in frame.columns]
    if not columns:
        return

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(15.2, 7.4))
    colors = [PALETTE["main"], PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    for column, color in zip(columns, colors):
        ax.plot(frame["date"], _zscore(frame[column]).rolling(14, min_periods=3).mean(), linewidth=2.0, color=color, label=column)
    ax.axhline(0, color=PALETTE["line"], linewidth=1.0, alpha=0.75)
    ax.set_ylabel("14-day smoothed z-score")
    ax.set_xlabel("Date")
    ax.set_title("Standardized Macro and Oil Market Indicators")
    ax.legend(frameon=False, ncol=min(len(columns), 5), loc="upper left")
    style_axis(ax, xgrid=False, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "chapter3_macro_standardized_lines.png")


def plot_modality_volume_lines() -> None:
    import matplotlib.pyplot as plt

    text_path = DATA_DIR / "chapter3_text_daily_summary.csv"
    image_path = DATA_DIR / "chapter3_image_daily_summary.csv"
    if not text_path.exists() or not image_path.exists():
        return
    text = pd.read_csv(text_path, parse_dates=["date"])
    image = pd.read_csv(image_path, parse_dates=["date"])
    frame = text[["date", "text_count"]].merge(image[["date", "image_count"]], on="date", how="outer").sort_values("date")
    for column in ["text_count", "image_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    configure_paper_style()
    fig, ax1 = plt.subplots(figsize=(15.2, 6.8))
    ax2 = ax1.twinx()
    ax1.plot(frame["date"], frame["text_count"].rolling(30, min_periods=3).mean(), color=PALETTE["blue"], linewidth=2.2, label="Text documents, 30D MA")
    ax2.plot(frame["date"], frame["image_count"].rolling(30, min_periods=3).mean(), color=PALETTE["green"], linewidth=2.2, label="Images, 30D MA")
    ax1.set_ylabel("Text documents")
    ax2.set_ylabel("Images")
    ax1.set_xlabel("Date")
    ax1.set_title("Daily Multimodal Asset Volume")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    style_axis(ax1, xgrid=False, ygrid=True)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "chapter3_text_image_volume_lines.png")


def plot_lag_profile_lines() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "lag_relationship_correlations.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame["target"].eq("target_residual_30d")].copy()
    if frame.empty:
        return
    best = pd.read_csv(TABLE_DIR / "lag_relationship_best_lags.csv")
    predictors = (
        best[best["target"].eq("target_residual_30d")]
        .dropna(subset=["abs_pearson_corr"])
        .sort_values("abs_pearson_corr", ascending=False)["predictor"]
        .head(7)
        .tolist()
    )
    frame = frame[frame["predictor"].isin(predictors)]

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    palette = [PALETTE["main"], PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"], PALETTE["mid"], PALETTE["gray"]]
    for predictor, color in zip(predictors, palette):
        subset = frame[frame["predictor"].eq(predictor)].sort_values("lag_days")
        ax.plot(subset["lag_days"], subset["pearson_corr"], marker="o", linewidth=2.0, markersize=5.5, color=color, label=predictor)
    ax.axhline(0, color=PALETTE["line"], linewidth=1.0)
    ax.set_xlabel("Lag days")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Lag Profiles for the 30-Day Brent Residual Target")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "lag_relationship_profile_lines.png")


def plot_rolling_fold_rmse_lines() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "robustness_fold_metrics.csv"
    comparison_path = TABLE_DIR / "robustness_comparison_table.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    roles = {}
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        roles = dict(zip(comp["model"], comp["role"]))

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    for model, group in frame.groupby("model", sort=False):
        role = roles.get(model, "baseline_control")
        ax.plot(group["fold"], group["rmse"], marker="o", linewidth=2.4 if role == "mainline" else 1.9, color=model_color(model, role), label=model)
    ax.set_xlabel("Rolling fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Fold-Level Rolling RMSE Trajectories")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "robustness_fold_rmse_lines.png")


def plot_rolling_fold_rmse_heatmap() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "robustness_fold_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    pivot = frame.pivot_table(index="model", columns="fold", values="rmse", aggfunc="mean")

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(11.6, 4.8 + 0.36 * len(pivot)))
    matrix = pivot.to_numpy(float)
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"F{int(item)}" for item in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9.4, color="white" if value > np.nanmax(matrix) * 0.55 else PALETTE["line"])
    ax.set_title("Fold-Level RMSE Heatmap")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("RMSE")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "robustness_fold_rmse_heatmap.png")


def plot_event_response_heatmap() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "event_window_response.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    data = frame.pivot_table(index="event_name", columns="window_days", values="post_minus_pre_brent_mean", aggfunc="mean")
    data = data.reindex(data.abs().max(axis=1).sort_values(ascending=False).index).head(16)
    labels = [_short_label(item, 48) for item in data.index]

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(10.4, 8.4))
    matrix = data.to_numpy(float)
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels([f"+/-{int(item)}d" for item in data.columns])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.4)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.8, color="white" if abs(value) > vmax * 0.55 else PALETTE["line"])
    ax.set_title("Event-Window Brent Mean Response Heatmap")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Post - pre Brent mean")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "event_window_brent_response_heatmap.png")


def plot_model_metric_3d_scatter() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "robustness_comparison_table.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path).dropna(subset=["test_rmse_mean", "rolling_score", "direction_acc_mean"])
    if frame.empty:
        return

    configure_paper_style()
    fig = plt.figure(figsize=(11.6, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in frame.iterrows():
        role = row.get("role", "baseline_control")
        color = model_color(row["model"], role)
        size = 85 if role != "mainline" else 150
        ax.scatter(row["test_rmse_mean"], row["rolling_score"], row["direction_acc_mean"] * 100.0, s=size, color=color, edgecolor=PALETTE["line"], linewidth=0.7)
        ax.text(row["test_rmse_mean"], row["rolling_score"], row["direction_acc_mean"] * 100.0, _short_label(row["model"], 24), fontsize=8.2)
    ax.set_xlabel("Fixed test RMSE", labelpad=10)
    ax.set_ylabel("Rolling score", labelpad=10)
    ax.set_zlabel("Direction accuracy (%)", labelpad=10)
    ax.set_title("3D Model Comparison: Error, Stability, and Direction Signal")
    ax.view_init(elev=24, azim=-45)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "robustness_model_metric_3d_scatter.png", dpi=200)


def plot_lag_correlation_3d_surface() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "lag_relationship_correlations.csv"
    best_path = TABLE_DIR / "lag_relationship_best_lags.csv"
    if not path.exists() or not best_path.exists():
        return
    frame = pd.read_csv(path)
    best = pd.read_csv(best_path)
    predictors = (
        best[best["target"].eq("target_residual_30d")]
        .dropna(subset=["abs_pearson_corr"])
        .sort_values("abs_pearson_corr", ascending=False)["predictor"]
        .head(8)
        .tolist()
    )
    data = frame[frame["target"].eq("target_residual_30d") & frame["predictor"].isin(predictors)]
    pivot = data.pivot_table(index="predictor", columns="lag_days", values="pearson_corr", aggfunc="mean").reindex(predictors)
    if pivot.empty:
        return
    matrix = pivot.astype(float).interpolate(axis=1, limit_direction="both").fillna(0.0).to_numpy()
    x_values = pivot.columns.to_numpy(float)
    y_values = np.arange(len(pivot.index), dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    configure_paper_style()
    fig = plt.figure(figsize=(12.6, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x_grid, y_grid, matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, linewidth=0.25, edgecolor="white", alpha=0.92)
    ax.set_xlabel("Lag days", labelpad=9)
    ax.set_ylabel("Predictor", labelpad=10)
    ax.set_zlabel("Pearson corr.", labelpad=9)
    ax.set_yticks(y_values)
    ax.set_yticklabels(pivot.index, fontsize=8.4)
    ax.set_title("3D Lag-Correlation Surface for 30-Day Brent Residual Target")
    ax.view_init(elev=28, azim=-58)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label="Pearson correlation")
    fig.subplots_adjust(left=0.03, right=0.88, top=0.92, bottom=0.05)
    save_figure(fig, FIGURE_DIR / "lag_relationship_3d_surface.png", dpi=200)


def main() -> None:
    ensure_dirs()
    plot_macro_standardized_lines()
    plot_modality_volume_lines()
    plot_lag_profile_lines()
    plot_rolling_fold_rmse_lines()
    plot_rolling_fold_rmse_heatmap()
    plot_event_response_heatmap()
    plot_model_metric_3d_scatter()
    plot_lag_correlation_3d_surface()
    print(f"Extended analysis figures refreshed in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
