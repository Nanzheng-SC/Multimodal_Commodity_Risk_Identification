from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
MODELING_ROOT = PROJECT_ROOT / "3_modeling"
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import metric_dict  # noqa: E402
from project_shared.paths import (  # noqa: E402
    EVENT_MANIFEST_PATH,
    IMAGE_DAILY_COVERAGE_CSV_PATH,
    STRUCTURED_DAILY_PATH,
    TEXT_DAILY_COVERAGE_CSV_PATH,
)
from project_shared.targets import REFERENCE_PRICE_COLUMN, compute_forward_average  # noqa: E402


DATA_DIR = ANALYSIS_ROOT / "data"
OUTPUT_DIR = ANALYSIS_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXPORT_ROOT = MODELING_ROOT / "results" / "export" / "daily_horizon30_late_gru_gate_mainline_final"
OFFICIAL_ROOT = MODELING_ROOT / "results" / "official" / "daily_horizon30"
ARIMA_ROOT = OFFICIAL_ROOT / "arima_residual" / "window_90"
HORIZON_DAYS = 30
MAINLINE_RUN_ID = "late_gru_gate_validation_selected_top2"

MODEL_DISPLAY_NAMES = {
    MAINLINE_RUN_ID: "TimeMixer (late.gru_gate)",
    "late.gru_concat": "TimeMixer (late.gru_concat)",
    "intermediate.gated": "TimeMixer (intermediate.gated)",
    "Structured": "TimeMixer (structured)",
    "Text": "TimeMixer (text)",
    "Image": "TimeMixer (image)",
    "Naive": "Naive",
    "HAR-no-leak": "HAR-no-leak",
    "LSTM": "LSTM",
    "ARIMA": "ARIMA",
}

PALETTE = {
    "main": "#174a68",
    "green": "#276749",
    "blue": "#2b6cb0",
    "orange": "#c46a1a",
    "red": "#b42318",
    "light": "#d1dbe2",
    "mid": "#7f9caf",
    "gray": "#64748b",
    "line": "#243442",
    "grid": "#dfe6ed",
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def display_model_name(run_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(run_id), str(run_id))


def add_display_model(frame: pd.DataFrame, model_column: str = "model") -> pd.DataFrame:
    result = frame.copy()
    if model_column in result.columns:
        if "run_id" not in result.columns:
            result.insert(0, "run_id", result[model_column].astype(str))
        result[model_column] = result[model_column].astype(str).map(display_model_name)
    return result


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "SimSun"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["line"],
            "xtick.color": PALETTE["line"],
            "ytick.color": PALETTE["line"],
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: Any, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    if xgrid:
        ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save_figure(fig: Any, path: Path, dpi: int = 190) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_structured_daily() -> pd.DataFrame:
    frame = pd.read_csv(STRUCTURED_DAILY_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["reference_brent"] = pd.to_numeric(
        frame["reference_brent"] if "reference_brent" in frame.columns else frame[REFERENCE_PRICE_COLUMN],
        errors="coerce",
    )
    frame["target_brent_avg_next_30d"] = compute_forward_average(
        pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce"),
        horizon=HORIZON_DAYS,
    )
    frame["target_residual_30d"] = frame["target_brent_avg_next_30d"] - frame["reference_brent"]
    frame["brent_return_1d"] = pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce").pct_change()
    frame["wti_return_1d"] = pd.to_numeric(frame.get("WTI"), errors="coerce").pct_change() if "WTI" in frame.columns else np.nan
    frame["brent_wti_spread"] = pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce") - pd.to_numeric(frame.get("WTI"), errors="coerce")
    frame["brent_rolling_mean_7"] = pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce").rolling(7, min_periods=2).mean()
    frame["brent_rolling_mean_30"] = pd.to_numeric(frame[REFERENCE_PRICE_COLUMN], errors="coerce").rolling(30, min_periods=5).mean()
    frame["brent_rolling_vol_7"] = frame["brent_return_1d"].rolling(7, min_periods=2).std()
    frame["brent_rolling_vol_30"] = frame["brent_return_1d"].rolling(30, min_periods=5).std()
    frame["abs_brent_return_1d"] = frame["brent_return_1d"].abs()
    threshold = frame["abs_brent_return_1d"].quantile(0.75)
    frame["high_volatility_flag"] = (frame["abs_brent_return_1d"] >= threshold).astype(int)
    return frame


def load_chapter3_structured() -> pd.DataFrame:
    path = DATA_DIR / "chapter3_structured_analysis.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
        return frame
    return load_structured_daily()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_event_windows() -> pd.DataFrame:
    path = DATA_DIR / "chapter3_event_windows.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["event_date", "window_7_start", "window_7_end", "window_30_start", "window_30_end"])
    events = read_jsonl(EVENT_MANIFEST_PATH)
    rows = []
    for event in events:
        event_date = pd.Timestamp(event.get("date"))
        tags = event.get("event_tags") or []
        rows.append(
            {
                "event_date": event_date,
                "event_name": event.get("title", ""),
                "event_type": ",".join(tags) if isinstance(tags, list) else str(tags),
                "window_7_start": event_date - pd.Timedelta(days=7),
                "window_7_end": event_date + pd.Timedelta(days=7),
                "window_30_start": event_date - pd.Timedelta(days=30),
                "window_30_end": event_date + pd.Timedelta(days=30),
            }
        )
    return pd.DataFrame(rows)


def available_prediction_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    main_path = EXPORT_ROOT / "tables" / "rolling_predictions.csv"
    if main_path.exists():
        main = pd.read_csv(main_path, parse_dates=["date"])
        if "model" in main.columns and not main.empty:
            for model, group in main.groupby("model"):
                frames[display_model_name(str(model))] = group.copy()
    arima_path = ARIMA_ROOT / "rolling_predictions.csv"
    if arima_path.exists():
        arima = pd.read_csv(arima_path, parse_dates=["date"])
        frames["ARIMA"] = arima.copy()
    return frames


def metrics_for_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    return metric_dict(
        frame["predicted_price"].to_numpy(np.float32),
        frame["target_price"].to_numpy(np.float32),
        frame["reference_brent"].to_numpy(np.float32),
    )


def write_manifest(path: Path, entries: Iterable[tuple[str, Path]]) -> None:
    payload = {key: project_relative(value) for key, value in entries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
