from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_common import (
    FIGURE_DIR,
    PALETTE,
    TABLE_DIR,
    configure_analysis_style,
    ensure_dirs,
    load_image_manifest,
    load_text_documents,
    save_figure,
    style_axis,
)


TOP_N = 20


def clean_string(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    return values.mask(values == "", fallback)


def build_text_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = load_text_documents()
    if text.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    text = text.copy()
    text["source_name"] = clean_string(text.get("source_name", pd.Series(index=text.index)), "Unknown")
    text["source_type"] = clean_string(text.get("source_type", pd.Series(index=text.index)), "Unknown")
    text["source_country"] = clean_string(text.get("source_country", pd.Series(index=text.index)), "Unknown")
    text["body_char_count"] = pd.to_numeric(text.get("body_char_count", 0), errors="coerce")
    text["open_source"] = text.get("open_source", False).fillna(False).astype(bool)
    text["has_text"] = text.get("has_text", False).fillna(False).astype(bool)
    text["has_summary"] = text.get("has_summary", False).fillna(False).astype(bool)
    total_docs = max(int(len(text)), 1)

    by_source = (
        text.groupby(["source_name", "source_type", "source_country"], dropna=False)
        .agg(
            document_count=("doc_id", "count"),
            active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())),
            first_date=("date", "min"),
            last_date=("date", "max"),
            mean_body_chars=("body_char_count", "mean"),
            open_source_count=("open_source", "sum"),
            has_text_rate=("has_text", "mean"),
            has_summary_rate=("has_summary", "mean"),
        )
        .reset_index()
    )
    by_source["share_of_documents"] = by_source["document_count"] / total_docs
    by_source["rank"] = by_source["document_count"].rank(method="first", ascending=False).astype(int)
    by_source = by_source.sort_values(["document_count", "source_name"], ascending=[False, True])

    by_type = (
        text.groupby("source_type", dropna=False)
        .agg(document_count=("doc_id", "count"), source_count=("source_name", "nunique"), active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())))
        .reset_index()
        .sort_values("document_count", ascending=False)
    )
    by_type["share_of_documents"] = by_type["document_count"] / total_docs

    by_country = (
        text.groupby("source_country", dropna=False)
        .agg(document_count=("doc_id", "count"), source_count=("source_name", "nunique"), active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())))
        .reset_index()
        .sort_values("document_count", ascending=False)
    )
    by_country["share_of_documents"] = by_country["document_count"] / total_docs
    return by_source, by_type, by_country


def build_image_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image = load_image_manifest()
    if image.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    image = image.copy()
    image["source_name"] = clean_string(image.get("source_name", pd.Series(index=image.index)), "Unknown")
    image["source_stream"] = clean_string(image.get("source_stream", pd.Series(index=image.index)), "Unknown")
    image["source_type"] = clean_string(image.get("source_type", pd.Series(index=image.index)), "Unknown")
    image["open_source"] = image.get("open_source", False).fillna(False).astype(bool)
    image["width"] = pd.to_numeric(image.get("width", np.nan), errors="coerce")
    image["height"] = pd.to_numeric(image.get("height", np.nan), errors="coerce")
    total_images = max(int(len(image)), 1)

    by_source = (
        image.groupby(["source_name", "source_stream", "source_type"], dropna=False)
        .agg(
            image_count=("image_id", "count"),
            active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())),
            first_date=("date", "min"),
            last_date=("date", "max"),
            open_source_count=("open_source", "sum"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
        )
        .reset_index()
    )
    by_source["share_of_images"] = by_source["image_count"] / total_images
    by_source["open_source_rate"] = by_source["open_source_count"] / by_source["image_count"].replace(0, np.nan)
    by_source["rank"] = by_source["image_count"].rank(method="first", ascending=False).astype(int)
    by_source = by_source.sort_values(["image_count", "source_name"], ascending=[False, True])

    by_stream = (
        image.groupby("source_stream", dropna=False)
        .agg(image_count=("image_id", "count"), source_count=("source_name", "nunique"), active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())))
        .reset_index()
        .sort_values("image_count", ascending=False)
    )
    by_stream["share_of_images"] = by_stream["image_count"] / total_images

    by_type = (
        image.groupby("source_type", dropna=False)
        .agg(image_count=("image_id", "count"), source_count=("source_name", "nunique"), active_days=("date", lambda values: int(pd.Series(values).dropna().dt.date.nunique())))
        .reset_index()
        .sort_values("image_count", ascending=False)
    )
    by_type["share_of_images"] = by_type["image_count"] / total_images
    return by_source, by_stream, by_type


def save_top_source_plot(frame: pd.DataFrame, value_col: str, share_col: str, label_col: str, title: str, path) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    data = frame.head(TOP_N).sort_values(value_col, ascending=True)
    configure_analysis_style()
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    colors = np.linspace(0.45, 0.9, len(data))
    ax.barh(data[label_col], data[value_col], color=plt.cm.Blues(colors), edgecolor=PALETTE["line"], linewidth=0.7)
    for idx, row in enumerate(data.itertuples(index=False)):
        value = getattr(row, value_col)
        share = getattr(row, share_col)
        ax.text(value + max(data[value_col]) * 0.012, idx, f"{int(value):,} ({share * 100:.1f}%)", va="center", fontsize=9.6)
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, data[value_col].max() * 1.18)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, path)


def save_stream_plot(frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    data = frame.sort_values("image_count", ascending=False)
    configure_analysis_style()
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    ax.bar(data["source_stream"], data["share_of_images"] * 100.0, color=PALETTE["green"], edgecolor=PALETTE["line"], linewidth=0.8)
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.text(idx, row.share_of_images * 100.0 + 1.0, f"{row.share_of_images * 100.0:.1f}%", ha="center", fontsize=10.2)
    ax.set_ylabel("Share of images (%)")
    ax.set_title("Image Source Stream Composition")
    ax.tick_params(axis="x", rotation=25)
    style_axis(ax, xgrid=False, ygrid=True)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "image_source_stream_composition.png")


def main() -> None:
    ensure_dirs()
    text_source, text_type, text_country = build_text_source_tables()
    image_source, image_stream, image_type = build_image_source_tables()

    text_source.to_csv(TABLE_DIR / "text_source_composition.csv", index=False, encoding="utf-8")
    text_type.to_csv(TABLE_DIR / "text_source_type_composition.csv", index=False, encoding="utf-8")
    text_country.to_csv(TABLE_DIR / "text_source_country_composition.csv", index=False, encoding="utf-8")
    image_source.to_csv(TABLE_DIR / "image_source_composition.csv", index=False, encoding="utf-8")
    image_stream.to_csv(TABLE_DIR / "image_source_stream_composition.csv", index=False, encoding="utf-8")
    image_type.to_csv(TABLE_DIR / "image_source_type_composition.csv", index=False, encoding="utf-8")

    save_top_source_plot(
        text_source,
        value_col="document_count",
        share_col="share_of_documents",
        label_col="source_name",
        title="Top Text Sources by Document Count",
        path=FIGURE_DIR / "text_source_composition_top20.png",
    )
    save_top_source_plot(
        image_source,
        value_col="image_count",
        share_col="share_of_images",
        label_col="source_name",
        title="Top Image Sources by Asset Count",
        path=FIGURE_DIR / "image_source_composition_top20.png",
    )
    save_stream_plot(image_stream)

    print(f"Wrote text source composition: {text_source.shape}")
    print(f"Wrote image source composition: {image_source.shape}")


if __name__ == "__main__":
    main()
