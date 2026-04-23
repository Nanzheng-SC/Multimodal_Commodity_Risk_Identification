# 5_statistical_analysis

This directory contains paper-facing statistical analysis assets. It does not train models.

Main roles:

- Chapter 3 descriptive and diagnostic data packages.
- Chapter 5 rolling robustness and event-window summaries.
- Final paper tables and figures used for writing and defense material.

Run the scripts in this order:

```powershell
python 5_statistical_analysis/scripts/build_chapter3_analysis_data.py
python 5_statistical_analysis/scripts/run_descriptive_stats.py
python 5_statistical_analysis/scripts/run_stationarity_tests.py
python 5_statistical_analysis/scripts/run_correlation_analysis.py
python 5_statistical_analysis/scripts/run_lag_relationship_analysis.py
python 5_statistical_analysis/scripts/run_source_composition.py
python 5_statistical_analysis/scripts/run_event_window_analysis.py
python 5_statistical_analysis/scripts/build_robustness_summary.py
python 5_statistical_analysis/scripts/plot_chapter5_figures.py
```

Final Chapter 5 figures:

| chapter section | figure |
| --- | --- |
| 5.1.1 fixed test comparison | `fixed_test_overview.png` |
| 5.1.2 rolling comparison | `rolling_overview_scatter.png` |
| 5.2.1 multimodal gain | `multimodal_gain_comparison.png` |
| 5.2.2 fusion strategy comparison | `fusion_strategy_comparison.png` |
| 5.3.1 high-risk period performance | `high_volatility_performance_comparison.png` |
| 5.3.2 mainline fold stability | `mainline_fold_stability.png` |

`plot_paper_figures.py` is kept as a compatibility notice only. It no longer
generates the old benchmark-context figure or calls the extended 3D/heatmap
diagnostic views.

Outputs:

```text
5_statistical_analysis/data/
5_statistical_analysis/outputs/tables/
5_statistical_analysis/outputs/figures/
```

The modeling baseline outputs stay under `3_modeling/`; this directory only organizes analysis assets.

`analysis_common.py` discovers the ARIMA official output directory from
`EXPORT_SUMMARY.json` first, then from ARIMA official metrics under
`3_modeling/results/official/daily_horizon30/`. Set `STAT_ANALYSIS_ARIMA_ROOT`
only when a manual override is needed.
