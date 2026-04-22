# ARIMA residual baseline

This directory contains the classical statistical baseline for the official daily horizon-30 task.

Task definition:

```text
target_residual_30d = target_brent_avg_next_30d - reference_brent
```

The implementation uses the existing `window_90` mainline split indices from:

```text
2_encoding_feature/outputs/daily/time_series_horizon30_mainline/late_gru_gate/window_90/
```

The model is intentionally single-variable ARIMA. It fits the historical residual sequence, forecasts future residuals, and restores price-level predictions as:

```text
predicted_price = reference_brent + predicted_residual
```

To keep the statistical baseline conservative, the runner applies a strict label-availability lag. A residual label dated `t` is only allowed in ARIMA history when its 30-day target window ends before the forecast origin. This prevents the baseline from using near-test residual labels that would require future Brent observations.

Run:

```powershell
python 3_modeling/baselines/arima/run_arima_benchmark.py --window-length 90 --update-export
```

Primary outputs:

```text
3_modeling/results/official/daily_horizon30/arima_residual/window_90/
```

The parent `arima_residual/` directory also receives a compatibility mirror of the key metric and prediction files.

ARIMAX is deliberately left as a future extension point. The current baseline does not use exogenous structured, text, image, or fusion features.
