from __future__ import annotations

import pandas as pd


DEFAULT_YEARS_BACK = 4
DEFAULT_END_DATE = "today"


def resolve_end_date(value: str | None = DEFAULT_END_DATE) -> str:
    raw = str(value or DEFAULT_END_DATE).strip().lower()
    if raw == "today":
        return pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def default_start_date_for_end(end_date: str, years_back: int = DEFAULT_YEARS_BACK) -> str:
    end_ts = pd.Timestamp(end_date).normalize()
    start_ts = (end_ts - pd.DateOffset(years=int(years_back)) + pd.Timedelta(days=1)).normalize()
    return start_ts.strftime("%Y-%m-%d")


def resolve_date_scope(
    start_date: str | None = None,
    end_date: str | None = DEFAULT_END_DATE,
    years_back: int = DEFAULT_YEARS_BACK,
) -> tuple[str, str]:
    resolved_end = resolve_end_date(end_date)
    if start_date and str(start_date).strip():
        resolved_start = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    else:
        resolved_start = default_start_date_for_end(resolved_end, years_back=years_back)
    if pd.Timestamp(resolved_start) > pd.Timestamp(resolved_end):
        raise ValueError(f"start_date must be <= end_date: {resolved_start} > {resolved_end}")
    return resolved_start, resolved_end


def date_scope_metadata(start_date: str, end_date: str, years_back: int = DEFAULT_YEARS_BACK) -> dict:
    total_days = len(pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D"))
    return {
        "start_date": start_date,
        "end_date": end_date,
        "years_back": int(years_back),
        "total_days": int(total_days),
    }
