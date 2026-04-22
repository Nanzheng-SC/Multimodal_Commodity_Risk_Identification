from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_common import FIGURE_DIR, PALETTE, TABLE_DIR, configure_paper_style, ensure_dirs, load_chapter3_structured, save_figure


CORRELATION_VARIABLES = [
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
    "target_residual_30d",
]


def save_heatmap(corr: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(12.8, 10.2))
    matrix = corr.to_numpy(float)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10.5)
    ax.set_yticklabels(corr.index, fontsize=10.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.6, color="white" if abs(value) > 0.55 else PALETTE["line"])
    ax.set_title("Pearson Correlation Matrix for Chapter 3 Variables", fontsize=16, pad=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=11)
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    ensure_dirs()
    frame = load_chapter3_structured()
    cols = [column for column in CORRELATION_VARIABLES if column in frame.columns]
    data = frame[cols].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    pearson.to_csv(TABLE_DIR / "correlation_pearson.csv", encoding="utf-8")
    spearman.to_csv(TABLE_DIR / "correlation_spearman.csv", encoding="utf-8")
    save_heatmap(pearson, FIGURE_DIR / "chapter3_correlation_heatmap.png")
    print(f"Wrote correlation matrices for {len(cols)} variables")


if __name__ == "__main__":
    main()
