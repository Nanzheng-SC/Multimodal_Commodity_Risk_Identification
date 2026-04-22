from __future__ import annotations

import pandas as pd

from analysis_common import FIGURE_DIR, PALETTE, TABLE_DIR, configure_paper_style, ensure_dirs, save_figure, style_axis


def plot_fixed_vs_rolling() -> None:
    import matplotlib.pyplot as plt

    comparison_path = TABLE_DIR / "robustness_comparison_table.csv"
    if not comparison_path.exists():
        return
    frame = pd.read_csv(comparison_path)
    if frame.empty or "test_rmse_mean" not in frame.columns:
        return
    frame = frame.dropna(subset=["test_rmse_mean", "rolling_score"]).sort_values("test_rmse_mean", ascending=False)
    configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    colors = [PALETTE["green"] if role == "mainline" else PALETTE["orange"] if model == "ARIMA" else PALETTE["light"] for model, role in zip(frame["model"], frame["role"])]
    axes[0].barh(frame["model"], frame["test_rmse_mean"], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
    axes[0].set_xlabel("Fixed test RMSE")
    axes[0].set_title("Fixed Test Error")
    axes[0].set_xlim(0, frame["test_rmse_mean"].max() * 1.14)
    style_axis(axes[0])

    roll = frame.sort_values("rolling_score", ascending=False)
    colors = [PALETTE["green"] if role == "mainline" else PALETTE["orange"] if model == "ARIMA" else PALETTE["light"] for model, role in zip(roll["model"], roll["role"])]
    axes[1].barh(roll["model"], roll["rolling_score"], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
    axes[1].set_xlabel("Rolling score")
    axes[1].set_title("6-Fold Rolling Stability")
    axes[1].set_xlim(0, roll["rolling_score"].max() * 1.14)
    style_axis(axes[1])
    fig.suptitle("Official Benchmark Context with Classical ARIMA Baseline", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, FIGURE_DIR / "paper_benchmark_context_with_arima.png")


def plot_high_volatility() -> None:
    import matplotlib.pyplot as plt

    path = TABLE_DIR / "high_volatility_model_performance.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    pivot = frame.pivot_table(index="model", columns="regime", values="rmse", aggfunc="mean").reset_index()
    if "high_volatility" not in pivot.columns:
        return
    pivot = pivot.sort_values("high_volatility", ascending=False)
    configure_paper_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    y = range(len(pivot))
    ax.barh(y, pivot["high_volatility"], color=PALETTE["red"], alpha=0.78, label="High volatility", edgecolor=PALETTE["line"], linewidth=0.7)
    if "normal_or_low_volatility" in pivot.columns:
        ax.scatter(pivot["normal_or_low_volatility"], y, color=PALETTE["blue"], s=90, label="Normal/low volatility", zorder=4, edgecolor=PALETTE["line"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot["model"])
    ax.set_xlabel("RMSE")
    ax.set_title("Model Performance in High-Volatility Days")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "high_volatility_performance_comparison.png")


def main() -> None:
    ensure_dirs()
    plot_fixed_vs_rolling()
    plot_high_volatility()
    print(f"Paper figures refreshed in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
