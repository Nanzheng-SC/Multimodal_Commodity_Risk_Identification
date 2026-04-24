from __future__ import annotations

from analysis_common import FIGURE_DIR


def main() -> None:
    print(
        "This compatibility entry no longer builds the evaluation figures. "
        "Run 5_statistical_analysis/scripts/plot_evaluation_figures.py instead. "
        f"Output directory: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
