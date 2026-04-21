from __future__ import annotations

import numpy as np
import pandas as pd


PRIMARY_TARGET_COLUMN = "target_brent_avg_next_7d"
AUX_TARGET_COLUMN = "target_brent_day7"
REFERENCE_PRICE_COLUMN = "Brent"
REFERENCE_ALIAS_COLUMN = "reference_brent"
STEP_RETURN_COLUMN = "brent_step_return"
STEP_VOLATILITY_COLUMN = "brent_step_volatility_7"
MARKET_CLOSED_COLUMN = "market_closed_flag"


def compute_forward_average(series: pd.Series, horizon: int = 7) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(len(values)):
        future = values[idx + 1 : idx + 1 + horizon]
        if len(future) == horizon and np.isfinite(future).all():
            output[idx] = float(np.mean(future))
    return pd.Series(output, index=series.index, dtype="float64")


def compute_forward_point(series: pd.Series, horizon: int = 7) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(len(values)):
        future_idx = idx + horizon
        if future_idx < len(values) and np.isfinite(values[future_idx]):
            output[idx] = float(values[future_idx])
    return pd.Series(output, index=series.index, dtype="float64")


def compute_step_return(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.pct_change().replace([np.inf, -np.inf], np.nan)


def compute_trailing_volatility(step_return: pd.Series, window: int = 7) -> pd.Series:
    return (
        pd.to_numeric(step_return, errors="coerce")
        .rolling(window=window, min_periods=2)
        .std()
        .replace([np.inf, -np.inf], np.nan)
    )


def build_price_targets(frame: pd.DataFrame, price_column: str = REFERENCE_PRICE_COLUMN) -> pd.DataFrame:
    result = frame.copy()
    result[REFERENCE_ALIAS_COLUMN] = pd.to_numeric(result[price_column], errors="coerce")
    result[STEP_RETURN_COLUMN] = compute_step_return(result[price_column])
    result[STEP_VOLATILITY_COLUMN] = compute_trailing_volatility(result[STEP_RETURN_COLUMN], window=7)
    result[PRIMARY_TARGET_COLUMN] = compute_forward_average(result[price_column], horizon=7)
    result[AUX_TARGET_COLUMN] = compute_forward_point(result[price_column], horizon=7)
    return result


def derive_month_end_rows(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    output = frame.copy()
    output[date_col] = pd.to_datetime(output[date_col], errors="coerce")
    output = output.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    output["_month"] = output[date_col].dt.to_period("M")
    month_end = output.groupby("_month", as_index=False).tail(1).copy()
    month_end["month"] = month_end[date_col].dt.strftime("%Y-%m")
    return month_end.drop(columns=["_month"])


def compute_forecast_window(anchor_value: str, frequency: str, horizon_days: int = 7) -> dict[str, str]:
    anchor = pd.Timestamp(anchor_value)
    start = anchor + pd.Timedelta(days=1)
    end = anchor + pd.Timedelta(days=horizon_days)
    return {
        "anchor": anchor.strftime("%Y-%m-%d"),
        "forecast_start_date": start.strftime("%Y-%m-%d"),
        "forecast_end_date": end.strftime("%Y-%m-%d"),
        "forecast_label": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "frequency": str(frequency),
    }

