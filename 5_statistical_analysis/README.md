# 5_statistical_analysis

This directory contains statistical analysis and evaluation assets. It does not train models.

Main roles:

- Descriptive and diagnostic data packages.
- Rolling robustness and event-window summaries.
- Evaluation tables and figures for reporting model behavior.

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
python 5_statistical_analysis/scripts/plot_evaluation_figures.py
```

Evaluation figures:

| purpose | figure |
| --- | --- |
| fixed test comparison | `fixed_test_overview.png` |
| rolling comparison | `rolling_overview_scatter.png` |
| multimodal gain | `multimodal_gain_comparison.png` |
| fusion strategy comparison | `fusion_strategy_comparison.png` |
| high-risk period performance | `high_volatility_performance_comparison.png` |
| mainline fold stability | `mainline_fold_stability.png` |

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
