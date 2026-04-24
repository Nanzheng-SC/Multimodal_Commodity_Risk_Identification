from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_common import (
    FIGURE_DIR,
    PALETTE,
    TABLE_DIR,
    configure_analysis_style,
    ensure_dirs,
    load_analysis_structured,
    load_event_windows,
    save_figure,
    style_axis,
)


def summarize_event_windows(structured: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    data = structured.copy()
    data["date"] = pd.to_datetime(data["date"])
    rows = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["event_date"])
        for window in (7, 30):
            start = event_date - pd.Timedelta(days=window)
            end = event_date + pd.Timedelta(days=window)
            before = data[(data["date"] >= start) & (data["date"] < event_date)]
            after = data[(data["date"] >= event_date) & (data["date"] <= end)]
            at_event = data[data["date"] == event_date]
            if before.empty or after.empty:
                continue
            rows.append(
                {
                    "event_date": event_date.date().isoformat(),
                    "event_name": event.get("event_name", ""),
                    "event_type": event.get("event_type", ""),
                    "window_days": window,
                    "pre_days": int(len(before)),
                    "post_days": int(len(after)),
                    "pre_brent_mean": float(pd.to_numeric(before["Brent"], errors="coerce").mean()),
                    "post_brent_mean": float(pd.to_numeric(after["Brent"], errors="coerce").mean()),
                    "post_minus_pre_brent_mean": float(pd.to_numeric(after["Brent"], errors="coerce").mean() - pd.to_numeric(before["Brent"], errors="coerce").mean()),
                    "post_abs_return_mean": float(pd.to_numeric(after.get("abs_brent_return_1d"), errors="coerce").mean()),
                    "event_target_residual_30d": float(pd.to_numeric(at_event.get("target_residual_30d"), errors="coerce").iloc[0]) if not at_event.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_event_plot(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    configure_analysis_style()
    frame = summary[summary["window_days"] == 30].copy()
    frame = frame.sort_values("post_abs_return_mean", ascending=True).tail(12)
    labels = frame["event_name"].astype(str).str.slice(0, 42)
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    colors = [PALETTE["red"] if value > 0 else PALETTE["blue"] for value in frame["post_minus_pre_brent_mean"]]
    ax.barh(labels, frame["post_minus_pre_brent_mean"], color=colors, edgecolor=PALETTE["line"], linewidth=0.8)
    ax.axvline(0, color=PALETTE["line"], linewidth=1.1)
    ax.set_xlabel("Post-window Brent mean - pre-window Brent mean")
    ax.set_title("Event Window Brent Response, 30-Day Window")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "event_window_brent_response.png")


def main() -> None:
    ensure_dirs()
    structured = load_analysis_structured()
    events = load_event_windows()
    summary = summarize_event_windows(structured, events)
    summary.to_csv(TABLE_DIR / "event_window_response.csv", index=False, encoding="utf-8")
    save_event_plot(summary)
    print(f"Wrote event-window response table: {summary.shape}")


if __name__ == "__main__":
    main()
