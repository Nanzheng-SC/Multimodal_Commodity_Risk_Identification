from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import pandas.util._decorators as _pd_decorators

_original_deprecate_kwarg = _pd_decorators.deprecate_kwarg


def _compat_deprecate_kwarg(*args, **kwargs):
    if args and isinstance(args[0], str):
        return _original_deprecate_kwarg(FutureWarning, *args, **kwargs)
    return _original_deprecate_kwarg(*args, **kwargs)


_pd_decorators.deprecate_kwarg = _compat_deprecate_kwarg

from statsmodels.tsa.stattools import adfuller, kpss

from analysis_common import TABLE_DIR, ensure_dirs, load_chapter3_structured


TEST_VARIABLES = [
    "Brent",
    "WTI",
    "USD_Index",
    "EPU",
    "GPR",
    "UAH_per_USD",
    "RUB_per_USD",
    "AED_per_USD",
    "brent_return_1d",
    "brent_rolling_vol_30",
    "target_residual_30d",
]


def run_adf(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 20 or clean.nunique() <= 1:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_lags": np.nan, "adf_nobs": len(clean), "adf_stationary_5pct": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, lags, nobs, *_ = adfuller(clean.to_numpy(float), autolag="AIC")
        return {
            "adf_stat": float(stat),
            "adf_pvalue": float(pvalue),
            "adf_lags": int(lags),
            "adf_nobs": int(nobs),
            "adf_stationary_5pct": bool(pvalue < 0.05),
        }
    except Exception:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_lags": np.nan, "adf_nobs": len(clean), "adf_stationary_5pct": False}


def run_kpss(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 20 or clean.nunique() <= 1:
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan, "kpss_lags": np.nan, "kpss_stationary_5pct": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, lags, _ = kpss(clean.to_numpy(float), regression="c", nlags="auto")
        return {
            "kpss_stat": float(stat),
            "kpss_pvalue": float(pvalue),
            "kpss_lags": int(lags),
            "kpss_stationary_5pct": bool(pvalue >= 0.05),
        }
    except Exception:
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan, "kpss_lags": np.nan, "kpss_stationary_5pct": False}


def main() -> None:
    ensure_dirs()
    frame = load_chapter3_structured()
    rows = []
    for variable in [item for item in TEST_VARIABLES if item in frame.columns]:
        row = {"variable": variable}
        row.update(run_adf(frame[variable]))
        row.update(run_kpss(frame[variable]))
        row["joint_interpretation"] = (
            "stationary"
            if row["adf_stationary_5pct"] and row["kpss_stationary_5pct"]
            else "mixed_or_nonstationary"
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "stationarity_tests.csv", index=False, encoding="utf-8")
    print(f"Wrote stationarity tests: {result.shape}")


if __name__ == "__main__":
    main()
