from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import EXPORT_ROOT, FIGURE_DIR, MAINLINE_RUN_ID, PALETTE, TABLE_DIR, configure_analysis_style, ensure_dirs, save_figure, style_axis


MAINLINE_LABEL = "TimeMixer (fusion; late.gru_gate)"
ARIMA_LABEL = "ARIMA"

FINAL_TEST_PATH = EXPORT_ROOT / "tables" / "final_test_leaderboard.csv"
FOLD_METRICS_PATH = EXPORT_ROOT / "tables" / "fold_metrics.csv"
ROBUSTNESS_PATH = TABLE_DIR / "robustness_comparison_table.csv"
HIGH_VOLATILITY_PATH = TABLE_DIR / "high_volatility_model_performance.csv"

MAIN_COLOR = PALETTE["main"]
ARIMA_COLOR = PALETTE["orange"]
NEUTRAL_COLOR = "#cfd8df"
NEUTRAL_DARK = "#7d8b96"
GRID_COLOR = PALETTE["grid"]


def configure_evaluation_style() -> None:
    import matplotlib.pyplot as plt

    configure_analysis_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.titlesize": 16,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def model_color(run_id: str, model: str | None = None) -> str:
    if str(run_id) == MAINLINE_RUN_ID or str(model) == MAINLINE_LABEL:
        return MAIN_COLOR
    return NEUTRAL_COLOR


def short_model_label(row: pd.Series) -> str:
    run_id = str(row.get("run_id", ""))
    model = str(row.get("model", ""))
    mapping = {
        MAINLINE_RUN_ID: "TimeMixer\n(fusion; late.gru_gate)",
        "late.gru_concat": "TimeMixer\nlate.gru_concat",
        "intermediate.gated": "TimeMixer\nintermediate.gated",
        "Structured": "TimeMixer\nstructured",
        "Text": "TimeMixer\ntext",
        "Image": "TimeMixer\nimage",
        "Naive": "Naive",
        "ARIMA": "ARIMA",
        "HAR-no-leak": "HAR-no-leak",
        "LSTM": "LSTM",
    }
    return mapping.get(run_id, model)


def add_bar_values(ax, values: pd.Series, offset: float | None = None) -> None:
    max_value = float(values.max()) if len(values) else 0.0
    offset = offset if offset is not None else max_value * 0.012
    for idx, value in enumerate(values):
        ax.text(float(value) + offset, idx, f"{float(value):.2f}", va="center", ha="left", fontsize=9.8, color=PALETTE["line"])


def plot_fixed_test_overview() -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(FINAL_TEST_PATH)
    require_columns(frame, ["run_id", "model", "test_rmse_mean", "test_mae_mean"], FINAL_TEST_PATH)
    frame = frame.sort_values("test_rmse_mean", ascending=True).reset_index(drop=True)
    labels = frame.apply(short_model_label, axis=1)
    colors = [model_color(row.run_id, row.model) for row in frame.itertuples(index=False)]
    y = np.arange(len(frame))

    configure_evaluation_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), sharey=True, gridspec_kw={"wspace": 0.10})
    metrics = [("test_rmse_mean", "RMSE"), ("test_mae_mean", "MAE")]
    for ax, (column, xlabel) in zip(axes, metrics):
        ax.barh(y, frame[column], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, float(frame[column].max()) * 1.14)
        add_bar_values(ax, frame[column])
        style_axis(ax)
    axes[0].invert_yaxis()
    axes[0].set_title("固定测试集 RMSE 排序")
    axes[1].set_title("固定测试集 MAE 排序")
    fig.suptitle("固定测试集综合比较", fontsize=17, y=0.98)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.88, bottom=0.12, wspace=0.12)
    save_figure(fig, FIGURE_DIR / "fixed_test_overview.png", dpi=220)


def plot_rolling_overview_scatter() -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(ROBUSTNESS_PATH)
    require_columns(frame, ["run_id", "model", "rolling_score", "direction_acc_mean", "role"], ROBUSTNESS_PATH)
    frame = frame.dropna(subset=["rolling_score", "direction_acc_mean"]).copy()
    frame["direction_pct"] = frame["direction_acc_mean"] * 100.0

    configure_evaluation_style()
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    for _, row in frame.iterrows():
        color = model_color(row["run_id"], row["model"])
        is_mainline = row["run_id"] == MAINLINE_RUN_ID
        ax.scatter(
            row["rolling_score"],
            row["direction_pct"],
            s=145 if is_mainline else 70,
            color=color,
            edgecolor=PALETTE["line"],
            linewidth=0.85,
            alpha=0.95 if is_mainline else 0.60,
            zorder=4 if is_mainline else 2,
        )

    labels_to_show = {MAINLINE_RUN_ID}
    for _, row in frame[frame["run_id"].isin(labels_to_show)].iterrows():
        dx = 0.035
        dy = 1.0
        ax.text(row["rolling_score"] + dx, row["direction_pct"] + dy, short_model_label(row).replace("\n", " "), fontsize=9.5, color=PALETTE["line"])

    ax.set_xlabel("Rolling score")
    ax.set_ylabel("方向准确率（%）")
    ax.set_title("Rolling 综合比较")
    ax.set_xlim(float(frame["rolling_score"].min()) - 0.15, float(frame["rolling_score"].max()) + 0.40)
    ax.set_ylim(max(0, float(frame["direction_pct"].min()) - 8), min(100, float(frame["direction_pct"].max()) + 8))
    style_axis(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "rolling_overview_scatter.png", dpi=220)


def merged_fixed_rolling() -> pd.DataFrame:
    fixed = pd.read_csv(FINAL_TEST_PATH)
    rolling = pd.read_csv(ROBUSTNESS_PATH)
    require_columns(fixed, ["run_id", "model", "test_rmse_mean"], FINAL_TEST_PATH)
    require_columns(rolling, ["run_id", "rolling_score"], ROBUSTNESS_PATH)
    fixed = fixed[["run_id", "model", "test_rmse_mean", "test_mae_mean"]].copy()
    rolling = rolling[["run_id", "rolling_score", "direction_acc_mean", "role"]].copy()
    return fixed.merge(rolling, on="run_id", how="left")


def plot_dual_panel_bars(frame: pd.DataFrame, labels: list[str], title: str, path: Path, highlight_run_ids: set[str]) -> None:
    import matplotlib.pyplot as plt

    y = np.arange(len(frame))
    colors = [model_color(row.run_id, row.model) if row.run_id in highlight_run_ids else NEUTRAL_COLOR for row in frame.itertuples(index=False)]

    configure_evaluation_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = [("test_rmse_mean", "固定测试集 RMSE"), ("rolling_score", "Rolling score")]
    for ax, (column, xlabel) in zip(axes, panels):
        ax.barh(y, frame[column], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, float(frame[column].max()) * 1.16)
        add_bar_values(ax, frame[column])
        style_axis(ax)
    axes[0].invert_yaxis()
    axes[0].set_title("固定测试集")
    axes[1].set_title("Rolling 结果")
    fig.suptitle(title, fontsize=16.5, y=0.98)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.14, wspace=0.12)
    save_figure(fig, path, dpi=220)


def plot_multimodal_gain_comparison() -> None:
    combined = merged_fixed_rolling()
    order = [MAINLINE_RUN_ID, "Image", "Text", "Structured"]
    frame = combined[combined["run_id"].isin(order)].copy()
    frame["order"] = frame["run_id"].map({run_id: idx for idx, run_id in enumerate(order)})
    frame = frame.sort_values("order").reset_index(drop=True)
    labels = ["Fusion\n(late.gru_gate)", "Image", "Text", "Structured"]
    plot_dual_panel_bars(
        frame,
        labels,
        "单模态与融合模型比较",
        FIGURE_DIR / "multimodal_gain_comparison.png",
        highlight_run_ids={MAINLINE_RUN_ID},
    )


def plot_fusion_strategy_comparison() -> None:
    combined = merged_fixed_rolling()
    order = [MAINLINE_RUN_ID, "late.gru_concat", "intermediate.gated"]
    frame = combined[combined["run_id"].isin(order)].copy()
    frame["order"] = frame["run_id"].map({run_id: idx for idx, run_id in enumerate(order)})
    frame = frame.sort_values("order").reset_index(drop=True)
    labels = ["late.gru_gate", "late.gru_concat", "intermediate.gated"]
    plot_dual_panel_bars(
        frame,
        labels,
        "融合策略比较",
        FIGURE_DIR / "fusion_strategy_comparison.png",
        highlight_run_ids={MAINLINE_RUN_ID},
    )


def plot_high_volatility_performance() -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(HIGH_VOLATILITY_PATH)
    require_columns(frame, ["model", "regime", "rmse", "direction_acc"], HIGH_VOLATILITY_PATH)
    frame = frame[frame["model"].isin([MAINLINE_LABEL, ARIMA_LABEL])].copy()
    if frame.empty:
        raise ValueError(f"{HIGH_VOLATILITY_PATH} has no mainline/ARIMA rows.")
    regime_order = ["normal_or_low_volatility", "high_volatility"]
    regime_labels = {"normal_or_low_volatility": "常态/低波动", "high_volatility": "高风险阶段"}
    model_order = [MAINLINE_LABEL, ARIMA_LABEL]
    model_labels = {MAINLINE_LABEL: "TimeMixer\n(fusion; late.gru_gate)", ARIMA_LABEL: "ARIMA"}
    bar_width = 0.34
    x = np.arange(len(model_order))

    configure_evaluation_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw={"wspace": 0.20})
    metric_info = [("rmse", "RMSE"), ("direction_acc", "方向准确率（%）")]
    for ax, (metric, ylabel) in zip(axes, metric_info):
        for offset_idx, regime in enumerate(regime_order):
            subset = frame[frame["regime"].eq(regime)].set_index("model").reindex(model_order)
            values = subset[metric].astype(float)
            if metric == "direction_acc":
                values = values * 100.0
            positions = x + (offset_idx - 0.5) * bar_width
            colors = [model_color(model, model) for model in model_order]
            alpha = 0.45 if regime == "normal_or_low_volatility" else 0.94
            hatch = "//" if regime == "normal_or_low_volatility" else ""
            ax.bar(positions, values, width=bar_width, color=colors, alpha=alpha, hatch=hatch, edgecolor=PALETTE["line"], linewidth=0.75, label=regime_labels[regime])
            for pos, value in zip(positions, values):
                ax.text(pos, float(value) + max(values) * 0.025, f"{float(value):.2f}", ha="center", va="bottom", fontsize=9.4)
        ax.set_xticks(x)
        ax.set_xticklabels([model_labels[model] for model in model_order])
        ax.set_ylabel(ylabel)
        ax.set_title("RMSE" if metric == "rmse" else "方向准确率")
        ax.set_ylim(0, max(float(frame[metric].max() * (100.0 if metric == "direction_acc" else 1.0)) * 1.25, 1.0))
        style_axis(ax, xgrid=False, ygrid=True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("高风险阶段表现", fontsize=16.5, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.16, wspace=0.24)
    save_figure(fig, FIGURE_DIR / "high_volatility_performance_comparison.png", dpi=220)


def plot_mainline_fold_stability() -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(FOLD_METRICS_PATH)
    require_columns(frame, ["run_id", "fold", "rmse", "direction_acc"], FOLD_METRICS_PATH)
    frame = frame[frame["run_id"].eq(MAINLINE_RUN_ID)].sort_values("fold").copy()
    if len(frame) != 6:
        raise ValueError(f"Expected 6 mainline folds in {FOLD_METRICS_PATH}, got {len(frame)}.")

    configure_evaluation_style()
    fig, ax1 = plt.subplots(figsize=(11.6, 6.4))
    ax2 = ax1.twinx()
    ax1.plot(frame["fold"], frame["rmse"], marker="o", linewidth=2.6, markersize=7.5, color=MAIN_COLOR, label="RMSE")
    ax2.plot(frame["fold"], frame["direction_acc"] * 100.0, marker="s", linewidth=2.2, markersize=6.8, color=ARIMA_COLOR, label="方向准确率")
    ax1.set_xlabel("Rolling 折次")
    ax1.set_ylabel("RMSE", color=MAIN_COLOR)
    ax2.set_ylabel("方向准确率（%）", color=ARIMA_COLOR)
    ax1.set_xticks(frame["fold"])
    ax1.set_ylim(0, float(frame["rmse"].max()) * 1.18)
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="y", colors=MAIN_COLOR)
    ax2.tick_params(axis="y", colors=ARIMA_COLOR)
    style_axis(ax1, xgrid=True, ygrid=True)
    ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    ax1.set_title("TimeMixer (fusion; late.gru_gate) 折次结果")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "mainline_fold_stability.png", dpi=220)


def main() -> None:
    ensure_dirs()
    plot_fixed_test_overview()
    plot_rolling_overview_scatter()
    plot_multimodal_gain_comparison()
    plot_fusion_strategy_comparison()
    plot_high_volatility_performance()
    plot_mainline_fold_stability()
    print(f"Evaluation figures refreshed in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
