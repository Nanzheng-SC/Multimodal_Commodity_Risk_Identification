from __future__ import annotations

import pandas as pd

from analysis_common import (
    DATA_DIR,
    EVENT_MANIFEST_PATH,
    EVENT_WINDOWS_PATH,
    IMAGE_DAILY_COVERAGE_CSV_PATH,
    IMAGE_DAILY_SUMMARY_PATH,
    STRUCTURED_ANALYSIS_PATH,
    TEXT_DAILY_COVERAGE_CSV_PATH,
    TEXT_DAILY_SUMMARY_PATH,
    ensure_dirs,
    load_structured_daily,
    read_jsonl,
)


def build_structured_package() -> pd.DataFrame:
    frame = load_structured_daily()
    keep = [
        "date",
        "Brent",
        "WTI",
        "USD_Index",
        "EPU",
        "GPR",
        "UAH_per_USD",
        "RUB_per_USD",
        "AED_per_USD",
        "reference_brent",
        "brent_step_return",
        "brent_step_volatility_7",
        "market_closed_flag",
        "brent_return_1d",
        "wti_return_1d",
        "brent_wti_spread",
        "brent_rolling_mean_7",
        "brent_rolling_mean_30",
        "brent_rolling_vol_7",
        "brent_rolling_vol_30",
        "abs_brent_return_1d",
        "high_volatility_flag",
        "target_brent_avg_next_30d",
        "target_residual_30d",
    ]
    keep = [column for column in keep if column in frame.columns]
    output = frame[keep].copy()
    output.to_csv(STRUCTURED_ANALYSIS_PATH, index=False, encoding="utf-8")
    return output


def build_text_summary() -> pd.DataFrame:
    text = pd.read_csv(TEXT_DAILY_COVERAGE_CSV_PATH)
    text["date"] = pd.to_datetime(text["date"]).dt.strftime("%Y-%m-%d")
    text["text_count"] = pd.to_numeric(text.get("doc_count", 0), errors="coerce").fillna(0).astype(int)
    text["text_covered"] = ((text["text_count"] > 0) & (pd.to_numeric(text.get("gap_flag", 0), errors="coerce").fillna(0) == 0)).astype(int)
    keep = [
        "date",
        "text_count",
        "text_covered",
        "gap_flag",
        "gap_reason",
        "primary_doc_id",
        "primary_title",
        "primary_source_name",
        "content_storage",
    ]
    output = text[[column for column in keep if column in text.columns]].copy()
    output.to_csv(TEXT_DAILY_SUMMARY_PATH, index=False, encoding="utf-8")
    return output


def build_image_summary() -> pd.DataFrame:
    image = pd.read_csv(IMAGE_DAILY_COVERAGE_CSV_PATH)
    image["date"] = pd.to_datetime(image["date"]).dt.strftime("%Y-%m-%d")
    image["image_count"] = pd.to_numeric(image.get("image_count", 0), errors="coerce").fillna(0).astype(int)
    image["image_covered"] = ((image["image_count"] > 0) & (pd.to_numeric(image.get("gap_flag", 0), errors="coerce").fillna(0) == 0)).astype(int)
    keep = [
        "date",
        "image_count",
        "image_covered",
        "gap_flag",
        "gap_reason",
        "primary_image_id",
        "primary_source_name",
        "primary_source_stream",
        "primary_thumbnail_path",
    ]
    output = image[[column for column in keep if column in image.columns]].copy()
    output.to_csv(IMAGE_DAILY_SUMMARY_PATH, index=False, encoding="utf-8")
    return output


def build_event_windows() -> pd.DataFrame:
    rows = []
    for event in read_jsonl(EVENT_MANIFEST_PATH):
        event_date = pd.Timestamp(event.get("date"))
        tags = event.get("event_tags") or []
        rows.append(
            {
                "event_date": event_date.date().isoformat(),
                "event_name": event.get("title", ""),
                "event_type": ",".join(tags) if isinstance(tags, list) else str(tags),
                "country_focus": ",".join(event.get("country_focus", [])) if isinstance(event.get("country_focus"), list) else event.get("country_focus", ""),
                "window_7_start": (event_date - pd.Timedelta(days=7)).date().isoformat(),
                "window_7_end": (event_date + pd.Timedelta(days=7)).date().isoformat(),
                "window_30_start": (event_date - pd.Timedelta(days=30)).date().isoformat(),
                "window_30_end": (event_date + pd.Timedelta(days=30)).date().isoformat(),
                "source_name": event.get("source_name", ""),
                "source_type": event.get("source_type", ""),
                "url": event.get("url", ""),
                "notes": event.get("notes", ""),
            }
        )
    output = pd.DataFrame(rows).sort_values("event_date")
    output.to_csv(EVENT_WINDOWS_PATH, index=False, encoding="utf-8")
    return output


def main() -> None:
    ensure_dirs()
    structured = build_structured_package()
    text = build_text_summary()
    image = build_image_summary()
    events = build_event_windows()
    print(f"Wrote structured package: {structured.shape}")
    print(f"Wrote text summary: {text.shape}")
    print(f"Wrote image summary: {image.shape}")
    print(f"Wrote event windows: {events.shape}")


if __name__ == "__main__":
    main()
