from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_common import (
    DATA_DIR,
    FIGURE_DIR,
    PALETTE,
    TABLE_DIR,
    configure_paper_style,
    ensure_dirs,
    load_chapter3_structured,
    save_figure,
    style_axis,
)


LAG_DAYS = [0, 1, 3, 7, 14, 21, 30, 45, 60, 90]
TARGETS = ["target_residual_30d", "brent_return_1d"]
PREDICTORS = [
    "Brent",
    "WTI",
    "USD_Index",
    "EPU",
    "GPR",
    "UAH_per_USD",
    "RUB_per_USD",
    "brent_return_1d",
    "brent_wti_spread",
    "brent_rolling_vol_7",
    "brent_rolling_vol_30",
    "text_count",
    "text_covered",
    "image_count",
    "image_covered",
]


def _corr_with_pvalue(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 30 or pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan, np.nan
    try:
        from scipy.stats import pearsonr, spearmanr

        if method == "spearman":
            result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1], nan_policy="omit")
            return float(result.statistic), float(result.pvalue)
        result = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)), np.nan


def build_lag_frame() -> pd.DataFrame:
    structured = load_chapter3_structured().copy()
    structured["date"] = pd.to_datetime(structured["date"])

    text_path = DATA_DIR / "chapter3_text_daily_summary.csv"
    if text_path.exists():
        text = pd.read_csv(text_path, parse_dates=["date"])[["date", "text_count", "text_covered"]]
        structured = structured.merge(text, on="date", how="left")

    image_path = DATA_DIR / "chapter3_image_daily_summary.csv"
    if image_path.exists():
        image = pd.read_csv(image_path, parse_dates=["date"])[["date", "image_count", "image_covered"]]
        structured = structured.merge(image, on="date", how="left")

    for column in ["text_count", "text_covered", "image_count", "image_covered"]:
        if column in structured.columns:
            structured[column] = pd.to_numeric(structured[column], errors="coerce").fillna(0.0)
    return structured.sort_values("date").reset_index(drop=True)


def run_lag_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [column for column in PREDICTORS if column in frame.columns]
    targets = [column for column in TARGETS if column in frame.columns]
    for target in targets:
        y = pd.to_numeric(frame[target], errors="coerce")
        for predictor in predictors:
            raw_x = pd.to_numeric(frame[predictor], errors="coerce")
            for lag in LAG_DAYS:
                x = raw_x.shift(int(lag))
                pearson, pearson_p = _corr_with_pvalue(x, y, "pearson")
                spearman, spearman_p = _corr_with_pvalue(x, y, "spearman")
                n = int(pd.concat([x, y], axis=1).dropna().shape[0])
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "lag_days": int(lag),
                        "n": n,
                        "pearson_corr": pearson,
                        "pearson_p_value": pearson_p,
                        "spearman_corr": spearman,
                        "spearman_p_value": spearman_p,
                        "interpretation": f"{predictor}(t-{lag}) vs {target}(t)",
                    }
                )
    return pd.DataFrame(rows)


def build_best_lag_table(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return correlations
    data = correlations.copy()
    data["abs_pearson_corr"] = data["pearson_corr"].abs()
    best = (
        data.sort_values(["target", "predictor", "abs_pearson_corr"], ascending=[True, True, False])
        .groupby(["target", "predictor"], as_index=False)
        .head(1)
        .sort_values(["target", "abs_pearson_corr"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return best


def save_lag_heatmap(correlations: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    target = "target_residual_30d"
    data = correlations[correlations["target"] == target].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="predictor", columns="lag_days", values="pearson_corr", aggfunc="mean")
    pivot = pivot.reindex([column for column in PREDICTORS if column in pivot.index])

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(13.8, 8.4))
    matrix = pivot.to_numpy(float)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(item)) for item in pivot.columns], fontsize=10.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(value) > 0.28 else PALETTE["line"])
    ax.set_xlabel("Predictor lag in days: x(t-lag) vs target(t)")
    ax.set_title("Lagged Pearson Correlations with 30-Day Brent Residual Target")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "lag_relationship_target_residual_heatmap.png")


def save_best_lag_plot(best: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    data = best[best["target"] == "target_residual_30d"].dropna(subset=["pearson_corr"]).copy()
    if data.empty:
        return
    data = data.sort_values("abs_pearson_corr", ascending=True).tail(12)

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = [PALETTE["blue"] if value >= 0 else PALETTE["orange"] for value in data["pearson_corr"]]
    ax.barh(data["predictor"], data["pearson_corr"], color=colors, edgecolor=PALETTE["line"], linewidth=0.75)
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.text(
            row.pearson_corr + (0.012 if row.pearson_corr >= 0 else -0.012),
            idx,
            f"lag {int(row.lag_days)}d",
            va="center",
            ha="left" if row.pearson_corr >= 0 else "right",
            fontsize=10.0,
        )
    ax.axvline(0, color=PALETTE["line"], linewidth=1.0)
    ax.set_xlabel("Best Pearson correlation by absolute value")
    ax.set_title("Strongest Lag Relationship for Each Predictor")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "lag_relationship_best_lags.png")


def main() -> None:
    ensure_dirs()
    frame = build_lag_frame()
    correlations = run_lag_correlations(frame)
    best = build_best_lag_table(correlations)

    correlations.to_csv(TABLE_DIR / "lag_relationship_correlations.csv", index=False, encoding="utf-8")
    best.to_csv(TABLE_DIR / "lag_relationship_best_lags.csv", index=False, encoding="utf-8")
    save_lag_heatmap(correlations)
    save_best_lag_plot(best)
    print(f"Wrote lag correlations: {correlations.shape}")
    print(f"Wrote best lag summary: {best.shape}")


if __name__ == "__main__":
    main()
