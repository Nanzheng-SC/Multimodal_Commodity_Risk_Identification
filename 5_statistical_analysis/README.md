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
python 5_statistical_analysis/scripts/run_event_window_analysis.py
python 5_statistical_analysis/scripts/build_robustness_summary.py
python 5_statistical_analysis/scripts/plot_paper_figures.py
```

Outputs:

```text
5_statistical_analysis/data/
5_statistical_analysis/outputs/tables/
5_statistical_analysis/outputs/figures/
```

The modeling baseline outputs stay under `3_modeling/`; this directory only organizes analysis assets.
