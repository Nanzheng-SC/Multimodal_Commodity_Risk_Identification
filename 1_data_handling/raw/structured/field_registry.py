from __future__ import annotations

from dataclasses import asdict, dataclass


FIELD_GROUPS = (
    "price_benchmarks",
    "macro_fx_uncertainty",
    "derived_features",
    "labels",
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: str
    source: str
    native_frequency: str
    fetch_method: str
    daily_fill_rule: str
    monthly_resample_rule: str
    modeling_enabled: bool
    notes: str = ""
    series_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

ACTIVE_FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        "Brent",
        "price_benchmarks",
        "FRED",
        "daily",
        "fred_series",
        "ffill_non_trading_days",
        "month_end_last",
        True,
        series_id="DCOILBRENTEU",
    ),
    FieldSpec(
        "WTI",
        "price_benchmarks",
        "FRED",
        "daily",
        "fred_series",
        "ffill_non_trading_days",
        "month_end_last",
        True,
        series_id="DCOILWTICO",
    ),
    FieldSpec(
        "USD_Index",
        "macro_fx_uncertainty",
        "FRED",
        "daily",
        "fred_series",
        "ffill_non_trading_days",
        "month_end_last",
        True,
        series_id="DTWEXBGS",
    ),
    FieldSpec(
        "EPU",
        "macro_fx_uncertainty",
        "FRED",
        "daily_7day",
        "fred_series",
        "ffill_non_trading_days",
        "month_end_last",
        True,
        series_id="USEPUINDXD",
    ),
    FieldSpec(
        "GPR",
        "macro_fx_uncertainty",
        "Iacoviello_GPR",
        "daily",
        "official_stata_download",
        "ffill_non_publication_days",
        "month_end_last",
        True,
        notes="Daily geopolitical risk series from the official GPR daily export.",
    ),
    FieldSpec(
        "UAH_per_USD",
        "macro_fx_uncertainty",
        "NBU",
        "daily",
        "nbu_exchange_site_json",
        "ffill_holidays_and_weekends",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "RUB_per_USD",
        "macro_fx_uncertainty",
        "CBR",
        "daily",
        "cbr_xml_dynamic",
        "ffill_holidays_and_weekends",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "AED_per_USD",
        "macro_fx_uncertainty",
        "CBUAE_USD_peg",
        "daily",
        "official_constant_series",
        "constant",
        "month_end_last",
        False,
        notes="Official AED/USD peg represented as a daily constant series.",
    ),
    FieldSpec(
        "reference_brent",
        "derived_features",
        "derived",
        "daily",
        "copy_from_brent",
        "same_day",
        "month_end_last",
        False,
    ),
    FieldSpec(
        "brent_step_return",
        "derived_features",
        "derived",
        "daily",
        "pct_change_from_brent",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "brent_step_volatility_7",
        "derived_features",
        "derived",
        "daily",
        "rolling_std_from_brent_return",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "brent_return_3d",
        "derived_features",
        "derived",
        "historical_daily",
        "pct_change_from_brent_3d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical Brent 3-day return; uses current and past prices only.",
    ),
    FieldSpec(
        "brent_return_7d",
        "derived_features",
        "derived",
        "historical_daily",
        "pct_change_from_brent_7d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical Brent 7-day return; uses current and past prices only.",
    ),
    FieldSpec(
        "brent_return_14d",
        "derived_features",
        "derived",
        "historical_daily",
        "pct_change_from_brent_14d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical Brent 14-day return; uses current and past prices only.",
    ),
    FieldSpec(
        "brent_return_30d",
        "derived_features",
        "derived",
        "historical_daily",
        "pct_change_from_brent_30d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical Brent 30-day return; uses current and past prices only.",
    ),
    FieldSpec(
        "brent_volatility_14d",
        "derived_features",
        "derived",
        "historical_daily",
        "rolling_std_from_brent_return_14d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical 14-day Brent return volatility.",
    ),
    FieldSpec(
        "brent_volatility_30d",
        "derived_features",
        "derived",
        "historical_daily",
        "rolling_std_from_brent_return_30d",
        "same_day",
        "month_end_last",
        True,
        notes="Historical 30-day Brent return volatility.",
    ),
    FieldSpec(
        "wti_brent_spread",
        "derived_features",
        "derived",
        "daily",
        "wti_minus_brent",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "usd_index_return_7d",
        "derived_features",
        "derived",
        "historical_daily",
        "pct_change_from_usd_index_7d",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "gpr_change_7d",
        "derived_features",
        "derived",
        "historical_daily",
        "change_from_gpr_7d",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "epu_change_7d",
        "derived_features",
        "derived",
        "historical_daily",
        "change_from_epu_7d",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "brent_zscore_30d",
        "derived_features",
        "derived",
        "historical_daily",
        "rolling_zscore_from_brent_30d",
        "same_day",
        "month_end_last",
        True,
    ),
    FieldSpec(
        "high_volatility_regime",
        "derived_features",
        "derived",
        "historical_daily",
        "rolling_volatility_quantile_flag",
        "same_day",
        "month_end_last",
        True,
        notes="One when historical 30-day Brent volatility is in the high-volatility regime.",
    ),
    FieldSpec(
        "fast_uptrend_regime",
        "derived_features",
        "derived",
        "historical_daily",
        "uptrend_quantile_flag",
        "same_day",
        "month_end_last",
        True,
        notes="One when historical 14-day Brent return indicates a fast uptrend.",
    ),
    FieldSpec(
        "market_closed_flag",
        "derived_features",
        "derived",
        "daily",
        "market_holiday_detector",
        "same_day",
        "month_end_last",
        False,
    ),
    FieldSpec(
        "target_brent_avg_next_7d",
        "labels",
        "derived",
        "daily",
        "forward_7d_average",
        "same_day",
        "month_end_anchor_forward_7d_average",
        False,
    ),
    FieldSpec(
        "target_brent_day7",
        "labels",
        "derived",
        "daily",
        "forward_day7_value",
        "same_day",
        "month_end_anchor_day7_value",
        False,
    ),
]


FIELD_SPECS = ACTIVE_FIELD_SPECS
FIELD_SPEC_BY_NAME = {item.name: item for item in ACTIVE_FIELD_SPECS}


def active_field_names() -> list[str]:
    return [item.name for item in ACTIVE_FIELD_SPECS]


def modeling_field_names() -> list[str]:
    return [item.name for item in ACTIVE_FIELD_SPECS if item.modeling_enabled]


def field_groups_for_names(names: list[str]) -> dict[str, list[str]]:
    grouped = {group: [] for group in FIELD_GROUPS}
    for name in names:
        spec = FIELD_SPEC_BY_NAME.get(name)
        if spec is None:
            continue
        grouped[spec.group].append(name)
    return {group: values for group, values in grouped.items() if values}
