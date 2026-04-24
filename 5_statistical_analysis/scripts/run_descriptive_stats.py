from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_common import (
    FIGURE_DIR,
    IMAGE_DAILY_SUMMARY_PATH,
    PALETTE,
    TABLE_DIR,
    TEXT_DAILY_SUMMARY_PATH,
    configure_analysis_style,
    ensure_dirs,
    load_analysis_structured,
    save_figure,
    style_axis,
)


KEY_VARIABLES = [
    "Brent",
    "WTI",
    "USD_Index",
    "EPU",
    "GPR",
    "UAH_per_USD",
    "RUB_per_USD",
    "AED_per_USD",
    "brent_return_1d",
    "brent_wti_spread",
    "brent_rolling_vol_30",
    "target_brent_avg_next_30d",
    "target_residual_30d",
]


def build_descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [item for item in KEY_VARIABLES if item in frame.columns]:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "count": int(values.notna().sum()),
                "missing_pct": float(values.isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_annual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year
    agg_cols = [column for column in ["Brent", "WTI", "USD_Index", "EPU", "GPR", "target_residual_30d", "brent_rolling_vol_30"] if column in data.columns]
    summary = data.groupby("year")[agg_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def build_coverage_summary() -> pd.DataFrame:
    rows = []
    if TEXT_DAILY_SUMMARY_PATH.exists():
        text = pd.read_csv(TEXT_DAILY_SUMMARY_PATH)
        rows.append(
            {
                "source": "text",
                "days": int(len(text)),
                "covered_days": int(pd.to_numeric(text["text_covered"], errors="coerce").fillna(0).sum()),
                "coverage_rate": float(pd.to_numeric(text["text_covered"], errors="coerce").fillna(0).mean()),
                "mean_daily_count": float(pd.to_numeric(text["text_count"], errors="coerce").mean()),
            }
        )
    if IMAGE_DAILY_SUMMARY_PATH.exists():
        image = pd.read_csv(IMAGE_DAILY_SUMMARY_PATH)
        rows.append(
            {
                "source": "image",
                "days": int(len(image)),
                "covered_days": int(pd.to_numeric(image["image_covered"], errors="coerce").fillna(0).sum()),
                "coverage_rate": float(pd.to_numeric(image["image_covered"], errors="coerce").fillna(0).mean()),
                "mean_daily_count": float(pd.to_numeric(image["image_count"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def save_price_residual_panel(frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    configure_analysis_style()
    data = frame.dropna(subset=["Brent", "target_residual_30d"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
    axes[0].plot(data["date"], data["Brent"], color=PALETTE["main"], linewidth=2.2, label="Brent spot/reference")
    if "target_brent_avg_next_30d" in data.columns:
        axes[0].plot(data["date"], data["target_brent_avg_next_30d"], color=PALETTE["orange"], linewidth=1.8, alpha=0.88, label="Future 30D average")
    axes[0].set_ylabel("USD/bbl")
    axes[0].set_title("Brent Price and 30-Day Forward Average")
    axes[0].legend(frameon=False, loc="upper right")
    style_axis(axes[0], xgrid=False, ygrid=True)

    axes[1].axhline(0.0, color=PALETTE["line"], linewidth=1.1)
    axes[1].fill_between(data["date"], data["target_residual_30d"], 0, color=PALETTE["green"], alpha=0.22)
    axes[1].plot(data["date"], data["target_residual_30d"], color=PALETTE["green"], linewidth=1.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Target Residual: Future 30D Average - Current Brent")
    style_axis(axes[1], xgrid=False, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "price_target_residual_panel.png")


def save_coverage_plot(coverage: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if coverage.empty:
        return
    configure_analysis_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = [PALETTE["blue"], PALETTE["green"]][: len(coverage)]
    ax.bar(coverage["source"], coverage["coverage_rate"] * 100.0, color=colors, edgecolor=PALETTE["line"], linewidth=0.8)
    for idx, row in enumerate(coverage.itertuples(index=False)):
        ax.text(idx, row.coverage_rate * 100.0 + 1.5, f"{row.coverage_rate * 100.0:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Covered days (%)")
    ax.set_title("Daily Text and Image Coverage")
    style_axis(ax, xgrid=False, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "modality_coverage.png")


def main() -> None:
    ensure_dirs()
    frame = load_analysis_structured()
    descriptive = build_descriptive_table(frame)
    annual = build_annual_summary(frame)
    coverage = build_coverage_summary()

    descriptive.to_csv(TABLE_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8")
    annual.to_csv(TABLE_DIR / "annual_structured_summary.csv", index=False, encoding="utf-8")
    coverage.to_csv(TABLE_DIR / "modality_coverage_summary.csv", index=False, encoding="utf-8")
    save_price_residual_panel(frame)
    save_coverage_plot(coverage)
    print(f"Wrote descriptive statistics: {descriptive.shape}")
    print(f"Wrote annual summary: {annual.shape}")
    print(f"Wrote coverage summary: {coverage.shape}")


if __name__ == "__main__":
    main()
