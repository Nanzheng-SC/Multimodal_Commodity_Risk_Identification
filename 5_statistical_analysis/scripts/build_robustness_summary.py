from __future__ import annotations

import pandas as pd

from analysis_common import (
    ARIMA_ROOT,
    EXPORT_ROOT,
    FIGURE_DIR,
    MAINLINE_RUN_ID,
    OFFICIAL_ROOT,
    PALETTE,
    TABLE_DIR,
    available_fixed_test_prediction_frames,
    display_model_name,
    available_prediction_frames,
    configure_analysis_style,
    ensure_dirs,
    load_analysis_structured,
    load_event_windows,
    metrics_for_prediction_frame,
    model_color,
    model_marker_size,
    save_figure,
    style_axis,
)


MAINLINE_NAME_FALLBACK = MAINLINE_RUN_ID


def load_rolling_leaderboard() -> pd.DataFrame:
    path = EXPORT_ROOT / "tables" / "rolling_leaderboard.csv"
    frame = pd.read_csv(path)
    if "rolling_score" not in frame.columns:
        frame["rolling_score"] = frame["rmse_mean"] + 0.25 * frame["rmse_std"].fillna(0.0)
    return frame.sort_values("rolling_score").reset_index(drop=True)


def load_fixed_leaderboard() -> pd.DataFrame:
    path = EXPORT_ROOT / "tables" / "final_test_leaderboard.csv"
    return pd.read_csv(path)


def build_comparison_table() -> pd.DataFrame:
    rolling = load_rolling_leaderboard()
    fixed = load_fixed_leaderboard()
    if "run_id" not in rolling.columns:
        rolling.insert(0, "run_id", rolling["model"].astype(str))
    if "run_id" not in fixed.columns:
        fixed.insert(0, "run_id", fixed["model"].astype(str))
    rolling = rolling.drop(columns=["model"]).rename(columns={"run_id": "model"})
    fixed = fixed.drop(columns=["model"]).rename(columns={"run_id": "model"})
    result = rolling.merge(fixed, on="model", how="left")
    role_map = {
        "ARIMA": "baseline_control",
        "Naive": "baseline_control",
        "HAR-no-leak": "baseline_control",
        "LSTM": "baseline_control",
        "Image": "single_modal_control",
        "Text": "single_modal_control",
        "Structured": "single_modal_control",
        "late.gru_concat": "fusioner_control",
        "intermediate.gated": "fusioner_control",
    }
    run_ids = set(result["model"].astype(str))
    mainline = MAINLINE_NAME_FALLBACK if MAINLINE_NAME_FALLBACK in run_ids else str(result.iloc[0]["model"])
    result["role"] = result["model"].map(role_map).fillna("mainline")
    result.loc[result["model"] == mainline, "role"] = "mainline"
    result = result.rename(columns={"model": "run_id"})
    result.insert(0, "model", result["run_id"].map(display_model_name))
    return result


def build_fold_metrics() -> pd.DataFrame:
    frames = []
    main_path = EXPORT_ROOT / "tables" / "fold_metrics.csv"
    if main_path.exists():
        frames.append(pd.read_csv(main_path))
    arima_path = ARIMA_ROOT / "fold_metrics.csv"
    if arima_path.exists():
        frames.append(pd.read_csv(arima_path))
    rolling_reference = OFFICIAL_ROOT.parent.parent / "rolling" / "daily_horizon30" / "robustness_6fold" / "tables" / "fold_metrics.csv"
    if rolling_reference.exists():
        frames.append(pd.read_csv(rolling_reference))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=[column for column in ["model", "fold"] if column in result.columns])
    if "model" in result.columns:
        if "run_id" not in result.columns:
            result.insert(0, "run_id", result["model"].astype(str))
        else:
            result["run_id"] = result["run_id"].fillna(result["model"]).astype(str)
        result["model"] = result["run_id"].astype(str).map(display_model_name)
    return result


def build_high_volatility_performance() -> pd.DataFrame:
    structured = load_analysis_structured()[["date", "reference_brent", "abs_brent_return_1d", "high_volatility_flag"]].copy()
    structured["date"] = pd.to_datetime(structured["date"])
    frames = available_fixed_test_prediction_frames(
        [
            display_model_name(MAINLINE_RUN_ID),
            "ARIMA",
            "Naive",
            "HAR-no-leak",
            "LSTM",
        ]
    )
    rows = []
    for model, pred in frames.items():
        merged = pred.copy()
        merged["date"] = pd.to_datetime(merged["date"])
        if "predicted_price" not in merged.columns and "y_pred" in merged.columns:
            merged["predicted_price"] = pd.to_numeric(merged["y_pred"], errors="coerce")
        if "target_price" not in merged.columns and "y_true" in merged.columns:
            merged["target_price"] = pd.to_numeric(merged["y_true"], errors="coerce")
        merged = merged.merge(structured, on="date", how="left")
        for flag, group in merged.groupby(merged["high_volatility_flag"].fillna(0).astype(int)):
            if group.empty:
                continue
            metrics = metrics_for_prediction_frame(group)
            rows.append(
                {
                    "model": model,
                    "regime": "high_volatility" if flag == 1 else "normal_or_low_volatility",
                    "n": int(len(group)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_event_window_model_performance() -> pd.DataFrame:
    events = load_event_windows()
    frames = available_prediction_frames()
    rows = []
    for model, pred in frames.items():
        pred = pred.copy()
        pred["date"] = pd.to_datetime(pred["date"])
        for _, event in events.iterrows():
            start = pd.Timestamp(event["window_30_start"])
            end = pd.Timestamp(event["window_30_end"])
            window = pred[(pred["date"] >= start) & (pred["date"] <= end)]
            if window.empty:
                continue
            metrics = metrics_for_prediction_frame(window)
            rows.append(
                {
                    "model": model,
                    "event_date": pd.Timestamp(event["event_date"]).date().isoformat(),
                    "event_name": event.get("event_name", ""),
                    "event_type": event.get("event_type", ""),
                    "n": int(len(window)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def save_rolling_score_plot(comparison: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if comparison.empty:
        return
    configure_analysis_style()
    frame = comparison.sort_values("rolling_score", ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    colors = [model_color(model, role) for model, role in zip(frame["model"], frame["role"])]
    ax.barh(frame["model"], frame["rolling_score"], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
    for idx, row in enumerate(frame.itertuples(index=False)):
        ax.text(row.rolling_score + 0.08, idx, f"{row.rolling_score:.3f}", va="center", fontsize=10.8)
    ax.set_xlabel("6-fold rolling score = RMSE mean + 0.25 * RMSE std")
    ax.set_title("Rolling Robustness Comparison Across Models")
    ax.set_xlim(0, max(frame["rolling_score"]) * 1.16)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "robustness_rolling_score_comparison.png")


def save_rmse_direction_scatter(comparison: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if comparison.empty or "direction_acc_mean" not in comparison.columns:
        return
    configure_analysis_style()
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    for _, row in comparison.iterrows():
        role = row.get("role", "")
        model = str(row["model"])
        color = model_color(model, role)
        size = model_marker_size(role)
        ax.scatter(row["rolling_score"], row["direction_acc_mean"] * 100.0, s=size, color=color, edgecolor=PALETTE["line"], linewidth=0.8, zorder=3)
        ax.text(row["rolling_score"] + 0.025, row["direction_acc_mean"] * 100.0 + 0.45, model, fontsize=9.5)
    ax.invert_xaxis()
    ax.set_xlabel("Rolling score, lower is better (axis inverted)")
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("Rolling Robustness: Error Stability vs Direction Signal")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "robustness_rmse_direction_scatter.png")


def save_event_performance_plot(event_perf: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    configure_analysis_style()
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    if event_perf.empty:
        ax.text(0.5, 0.5, "No event windows overlap available rolling prediction dates", ha="center", va="center", fontsize=14, color=PALETTE["line"])
        ax.set_axis_off()
        save_figure(fig, FIGURE_DIR / "event_window_performance.png")
        return
    frame = event_perf.groupby("model", as_index=False).agg(rmse=("rmse", "mean"), mae=("mae", "mean"), n=("n", "sum"))
    frame = frame.sort_values("rmse", ascending=False)
    mainline_display = display_model_name(MAINLINE_NAME_FALLBACK)
    colors = [
        model_color(model, "mainline" if str(model) == mainline_display else "baseline_control")
        for model in frame["model"]
    ]
    ax.barh(frame["model"], frame["rmse"], color=colors, edgecolor=PALETTE["line"], linewidth=0.8)
    for idx, row in enumerate(frame.itertuples(index=False)):
        ax.text(row.rmse + 0.06, idx, f"{row.rmse:.2f} (n={row.n})", va="center", fontsize=10.8)
    ax.set_xlabel("Mean event-window RMSE")
    ax.set_title("Model Performance Around Event Windows")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "event_window_performance.png")


def main() -> None:
    ensure_dirs()
    comparison = build_comparison_table()
    fold_metrics = build_fold_metrics()
    high_vol = build_high_volatility_performance()
    event_perf = build_event_window_model_performance()

    comparison.to_csv(TABLE_DIR / "robustness_comparison_table.csv", index=False, encoding="utf-8")
    fold_metrics.to_csv(TABLE_DIR / "robustness_fold_metrics.csv", index=False, encoding="utf-8")
    high_vol.to_csv(TABLE_DIR / "high_volatility_model_performance.csv", index=False, encoding="utf-8")
    event_perf.to_csv(TABLE_DIR / "event_window_model_performance.csv", index=False, encoding="utf-8")

    save_rolling_score_plot(comparison)
    save_rmse_direction_scatter(comparison)
    save_event_performance_plot(event_perf)
    print(f"Wrote robustness comparison: {comparison.shape}")
    print(f"Wrote robustness fold metrics: {fold_metrics.shape}")
    print(f"Wrote high-volatility performance: {high_vol.shape}")
    print(f"Wrote event-window model performance: {event_perf.shape}")


if __name__ == "__main__":
    main()
