from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "4_decision_rl"
EXPORT_ROOT = PROJECT_ROOT / "3_modeling" / "results" / "export" / "daily_horizon30_late_gru_gate_mainline_final"
OFFICIAL_ROOT = PROJECT_ROOT / "3_modeling" / "results" / "official" / "daily_horizon30"
ANALYSIS_OUTPUT_ROOT = PROJECT_ROOT / "5_statistical_analysis" / "outputs"
ANALYSIS_TABLES = ANALYSIS_OUTPUT_ROOT / "tables"
ANALYSIS_FIGURES = ANALYSIS_OUTPUT_ROOT / "figures"
STRUCTURED_DAILY = PROJECT_ROOT / "1_data_handling" / "raw" / "structured" / "structured_daily_merged.csv"

DASHBOARD_JSON_PATH = DASHBOARD_DIR / "decision_dashboard_data.json"
DASHBOARD_JS_PATH = DASHBOARD_DIR / "decision_dashboard_data.js"

MAINLINE_MODEL = "late_gru_gate_validation_selected_top2"
PREDICTION_PATHS = {
    "TimeMixer-Fusion": OFFICIAL_ROOT
    / "timemixer_late_gru_gate_mainline_final"
    / "window_validation_selected"
    / "best_run"
    / "predictions_test.csv",
    "ARIMA": OFFICIAL_ROOT / "arima_residual" / "window_90" / "best_run" / "predictions_test.csv",
    "Naive": OFFICIAL_ROOT / "naive_reference" / "window_90" / "best_run" / "predictions_test.csv",
    "HAR-no-leak": OFFICIAL_ROOT / "har_no_leak_residual" / "window_30" / "best_run" / "predictions_test.csv",
    "LSTM": OFFICIAL_ROOT / "lstm_residual" / "window_90" / "best_run" / "predictions_test.csv",
}

EVALUATION_FIGURES = [
    ("fixed_test_overview", "固定测试集比较", "fixed_test_overview.png"),
    ("rolling_overview_scatter", "Rolling 综合比较", "rolling_overview_scatter.png"),
    ("multimodal_gain_comparison", "单模态与多模态对比", "multimodal_gain_comparison.png"),
    ("fusion_strategy_comparison", "融合策略比较", "fusion_strategy_comparison.png"),
    ("high_volatility_performance", "高风险阶段表现", "high_volatility_performance_comparison.png"),
    ("mainline_fold_stability", "主模型折次稳定性", "mainline_fold_stability.png"),
    ("mainline_fold_stability_bars", "主模型折次稳定性柱状图", "mainline_fold_stability_bars.png"),
]

INDUSTRY_PROFILES = [
    {
        "industry": "航空与航运",
        "icon": "✈",
        "exposure_type": "航煤 / 船燃成本",
        "sensitivity": "高",
        "exposure_multiplier": 1.35,
        "basis_multiplier": 0.72,
        "hedge_efficiency": 0.66,
        "transmission": "Brent 上行通常经航煤、船燃和燃油附加费传导到现金成本，运价或票价传导滞后时会直接压缩利润。",
        "hedging_focus": "优先锁定未来 1-2 个季度核心燃油敞口，并将燃油附加费、采购节奏和套保头寸联动管理。",
        "policy_focus": "交通运输相关行业应建立燃油成本传导和套保执行的月度联动机制。",
        "defaults": {"annualRevenue": 200000, "energyCostShare": 30, "netMargin": 8, "debtRatio": 58, "existingHedgeRatio": 20},
    },
    {
        "industry": "物流与公路运输",
        "icon": "🚚",
        "exposure_type": "柴油成本",
        "sensitivity": "中高",
        "exposure_multiplier": 1.20,
        "basis_multiplier": 0.45,
        "hedge_efficiency": 0.62,
        "transmission": "Brent 变动通过柴油批发价和燃油附加费影响单位运输成本，客户合同调价周期会放大短期现金流压力。",
        "hedging_focus": "结合运价调整周期动态配置套保比例，重点监控燃料成本对毛利的挤压。",
        "policy_focus": "建议将燃油附加费条款、调价周期和成本指数挂钩机制纳入经营合同管理。",
        "defaults": {"annualRevenue": 80000, "energyCostShare": 22, "netMargin": 12, "debtRatio": 52, "existingHedgeRatio": 15},
    },
    {
        "industry": "化工与塑料",
        "icon": "⚗",
        "exposure_type": "石脑油 / 烯烃原料",
        "sensitivity": "高",
        "exposure_multiplier": 1.10,
        "basis_multiplier": 0.58,
        "hedge_efficiency": 0.58,
        "transmission": "原油价格通过石脑油、PX、乙烯、丙烯等链条影响原料成本和库存重估，下游产品价格传导通常存在滞后。",
        "hedging_focus": "将高风险月份与采购节奏、库存周转和产品价差管理联动，避免原料上涨挤压加工利润。",
        "policy_focus": "化工企业应结合库存周期和订单锁价安排分层套保，避免单点集中执行。",
        "defaults": {"annualRevenue": 100000, "energyCostShare": 18, "netMargin": 15, "debtRatio": 45, "existingHedgeRatio": 10},
    },
    {
        "industry": "炼化企业",
        "icon": "🏭",
        "exposure_type": "原油采购 / 裂解价差",
        "sensitivity": "双向",
        "exposure_multiplier": 1.00,
        "basis_multiplier": 0.86,
        "hedge_efficiency": 0.64,
        "transmission": "原油上涨抬升采购成本；若成品油和化工品价格传导滞后，裂解价差和库存价值会同步承压。",
        "hedging_focus": "将采购、库存、加工利润和套保头寸一体化管理，同时关注绝对价格风险和 crack spread 风险。",
        "policy_focus": "建议建立原油库存、产品价差和套保头寸的统一风险看板。",
        "defaults": {"annualRevenue": 150000, "energyCostShare": 12, "netMargin": 10, "debtRatio": 55, "existingHedgeRatio": 25},
    },
    {
        "industry": "油气上游",
        "icon": "⛽",
        "exposure_type": "销售收入",
        "sensitivity": "反向",
        "exposure_multiplier": 1.05,
        "basis_multiplier": -0.92,
        "hedge_efficiency": 0.60,
        "transmission": "Brent 下行会直接压缩销售收入、储量价值和资本开支能力；上行则改善收入但也可能带来预算和投资节奏变化。",
        "hedging_focus": "重点管理下行收入保护，可结合保护性看跌、collar 或分层远期销售结构。",
        "policy_focus": "上游企业应将油价下行情景纳入资本开支、现金流和债务覆盖率压力测试。",
        "defaults": {"annualRevenue": 180000, "energyCostShare": 6, "netMargin": 22, "debtRatio": 48, "existingHedgeRatio": 18},
    },
]

SCENARIOS = {
    "base": {"label": "基准情景", "factor": 1.00, "note": "采用当前模型风险信号。"},
    "mild": {"label": "温和情景", "factor": 0.72, "note": "假设价格风险传导弱于当前信号。"},
    "stress": {"label": "压力情景", "factor": 1.55, "note": "假设价格风险传导强于当前信号。"},
    "custom": {"label": "自定义情景", "factor": 1.00, "note": "使用用户输入的基础风险值。"},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def load_structured() -> pd.DataFrame:
    frame = pd.read_csv(STRUCTURED_DAILY, parse_dates=["date"])
    needed = ["date", "reference_brent", "Brent", "WTI", "USD_Index", "EPU", "GPR", "market_closed_flag"]
    return frame[[column for column in needed if column in frame.columns]].copy()


def load_prediction_series(structured: pd.DataFrame) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = {}
    reference = structured[["date", "reference_brent"]].copy()
    for label, path in PREDICTION_PATHS.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"]).merge(reference, on="date", how="left")
        frame["predicted_residual"] = frame["y_pred"] - frame["reference_brent"]
        frame["actual_residual"] = frame["y_true"] - frame["reference_brent"]
        frame["direction_hit"] = frame.apply(
            lambda row: sign(float(row["predicted_residual"])) == sign(float(row["actual_residual"])),
            axis=1,
        )
        series[label] = [
            {
                "date": row["date"].date().isoformat(),
                "reference_brent": safe_float(row["reference_brent"]),
                "actual_30d_avg": safe_float(row["y_true"]),
                "predicted_30d_avg": safe_float(row["y_pred"]),
                "predicted_residual": safe_float(row["predicted_residual"]),
                "actual_residual": safe_float(row["actual_residual"]),
                "error": safe_float(row["error"]),
                "abs_error": safe_float(row["abs_error"]),
                "direction_hit": bool(row["direction_hit"]),
            }
            for _, row in frame.iterrows()
        ]
    return series


def build_leaderboards() -> tuple[list[dict], list[dict], list[dict]]:
    test = pd.read_csv(EXPORT_ROOT / "tables" / "final_test_leaderboard.csv")
    robustness_path = ANALYSIS_TABLES / "robustness_comparison_table.csv"
    rolling = pd.read_csv(robustness_path if robustness_path.exists() else EXPORT_ROOT / "tables" / "rolling_leaderboard.csv")
    selection = pd.read_csv(EXPORT_ROOT / "tables" / "final_selection_basis.csv")
    return (
        test.to_dict(orient="records"),
        rolling.to_dict(orient="records"),
        selection.to_dict(orient="records"),
    )


def build_recent_market(structured: pd.DataFrame) -> dict:
    recent = structured.tail(45).copy()
    latest = structured.iloc[-1]
    brent = pd.to_numeric(structured["reference_brent"], errors="coerce")
    return {
        "latest_date": latest["date"].date().isoformat(),
        "latest_reference_brent": safe_float(latest["reference_brent"]),
        "latest_wti": safe_float(latest.get("WTI", 0.0)),
        "latest_usd_index": safe_float(latest.get("USD_Index", 0.0)),
        "brent_30d_change": safe_float(brent.iloc[-1] - brent.iloc[-31]) if len(brent) > 31 else 0.0,
        "brent_30d_change_pct": safe_float(brent.iloc[-1] / brent.iloc[-31] - 1.0) if len(brent) > 31 else 0.0,
        "history": [
            {
                "date": row["date"].date().isoformat(),
                "reference_brent": safe_float(row["reference_brent"]),
                "WTI": safe_float(row.get("WTI", 0.0)),
                "USD_Index": safe_float(row.get("USD_Index", 0.0)),
                "market_closed_flag": int(row.get("market_closed_flag", 0)),
            }
            for _, row in recent.iterrows()
        ],
    }


def build_distribution(rolling_predictions: pd.DataFrame) -> dict:
    frame = rolling_predictions.copy()
    frame["abs_residual_pct"] = (frame["target_residual"] / frame["reference_brent"].clip(lower=1e-6)).abs()
    values = pd.to_numeric(frame["abs_residual_pct"], errors="coerce").dropna()
    if values.empty:
        values = pd.Series([0.01, 0.02, 0.04, 0.06])
    return {
        "count": int(values.shape[0]),
        "mean": safe_float(values.mean()),
        "q25": safe_float(values.quantile(0.25)),
        "q50": safe_float(values.quantile(0.50)),
        "q75": safe_float(values.quantile(0.75)),
        "q90": safe_float(values.quantile(0.90)),
        "max": safe_float(values.max()),
    }


def build_thresholds(distribution: dict) -> dict:
    return {
        "low_upper": distribution["q25"],
        "watch_upper": distribution["q50"],
        "elevated_upper": distribution["q75"],
        "high_reference": distribution["q90"],
    }


def classify_risk(predicted_residual: float, reference: float, thresholds: dict) -> dict:
    pct = predicted_residual / max(reference, 1e-6)
    abs_pct = abs(pct)
    if abs_pct <= thresholds["low_upper"]:
        key, label = "low", "低风险"
    elif abs_pct <= thresholds["watch_upper"]:
        key, label = "watch", "关注风险"
    elif abs_pct <= thresholds["elevated_upper"]:
        key, label = "elevated", "较高风险"
    else:
        key, label = "high", "高风险"
    direction = "upside_cost_pressure" if predicted_residual > 0 else "downside_revenue_pressure" if predicted_residual < 0 else "neutral"
    return {
        "direction": direction,
        "level_key": key,
        "level_label": label,
        "predicted_residual_pct": pct,
        "base_risk_value": abs_pct,
    }


def read_optional_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def build_figures() -> list[dict]:
    figures = []
    for figure_id, title, filename in EVALUATION_FIGURES:
        path = ANALYSIS_FIGURES / filename
        if path.exists():
            figures.append({"id": figure_id, "title": title, "path": rel(path)})
    return figures


def build_payload() -> dict:
    summary = read_json(EXPORT_ROOT / "EXPORT_SUMMARY.json")
    structured = load_structured()
    prediction_series = load_prediction_series(structured)
    main_series = prediction_series["TimeMixer-Fusion"]
    latest_eval = main_series[-1]
    test_leaderboard, rolling_leaderboard, selection_basis = build_leaderboards()

    rolling_predictions = pd.read_csv(EXPORT_ROOT / "tables" / "rolling_predictions.csv")
    market_focus_days = pd.read_csv(EXPORT_ROOT / "tables" / "market_focus_days.csv")
    fold_metrics = pd.read_csv(EXPORT_ROOT / "tables" / "fold_metrics.csv")
    distribution = build_distribution(rolling_predictions)
    thresholds = build_thresholds(distribution)
    risk_signal = classify_risk(latest_eval["predicted_residual"], latest_eval["reference_brent"], thresholds)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dashboard_title": "Enterprise Brent Hedging Risk Console",
        "project_note": "面向企业燃料、原料、库存与收入风险管理的 Brent 30 日价格风险控制台。",
        "source_note": "模型信号来自最终 TimeMixer + fusion + late.gru_gate 主线；风险阈值来自 rolling 评估窗口的历史残差分布。",
        "data_scope": {
            "start": "2022-04-17",
            "end": "2026-04-16",
            "days": 1461,
            "evaluation_test_days": len(main_series),
            "rolling_folds": int(summary["rolling_metrics"]["folds"]),
        },
        "task": {
            "target": "future_30d_brent_average_residual",
            "target_definition": "target_brent_avg_next_30d - reference_brent",
            "forecast_horizon_days": int(summary["official_metrics"]["forecast_horizon_days"]),
            "frequency": "daily",
        },
        "model": {
            "name": summary["official_metrics"]["model"],
            "input_variant": summary["official_metrics"]["input_variant"],
            "fusion_method": summary["selected_method"],
            "final_name": summary["final_name"],
            "validation_status": summary["official_metrics"].get("validation_status", "PASS"),
            "deployment_mode": summary["official_metrics"]["deployment_mode"],
            "selection_rule": summary["official_metrics"]["selection_rule"],
            "ensemble_member_count": len(summary["official_metrics"].get("selected_members", [])),
        },
        "latest_market": build_recent_market(structured),
        "latest_evaluation": latest_eval,
        "model_forecast": {
            "forecast_date": latest_eval["date"],
            "reference_brent": latest_eval["reference_brent"],
            "predicted_30d_avg": latest_eval["predicted_30d_avg"],
            "predicted_residual": latest_eval["predicted_residual"],
            "predicted_residual_pct": risk_signal["predicted_residual_pct"],
            "base_risk_value": risk_signal["base_risk_value"],
            "risk_level": risk_signal["level_label"],
            "risk_level_key": risk_signal["level_key"],
            "risk_direction": risk_signal["direction"],
        },
        "risk_signal": risk_signal,
        "historical_distribution": distribution,
        "risk_level_thresholds": thresholds,
        "risk_level_labels": {
            "low": "低风险",
            "watch": "关注风险",
            "elevated": "较高风险",
            "high": "高风险",
        },
        "scenarios": SCENARIOS,
        "official_metrics": {
            "test_rmse": safe_float(summary["official_metrics"]["deployed_run"]["test_rmse"]),
            "test_mae": safe_float(summary["official_metrics"]["deployed_run"]["test_mae"]),
            "test_mape": safe_float(summary["official_metrics"]["deployed_run"]["test_mape"]),
            "test_direction_acc": safe_float(summary["official_metrics"]["deployed_run"]["direction_acc"]),
            "rolling_score": safe_float(summary["rolling_metrics"]["rolling_score"]),
            "rolling_rmse_mean": safe_float(summary["rolling_metrics"]["rmse_mean"]),
            "rolling_rmse_std": safe_float(summary["rolling_metrics"]["rmse_std"]),
            "rolling_direction_acc": safe_float(summary["rolling_metrics"]["direction_acc_mean"]),
            "best_control_test_rmse": safe_float(summary["victory_check"]["best_control_test_rmse"]),
            "best_control_rolling_score": safe_float(summary["victory_check"]["best_control_rolling_score"]),
        },
        "leaderboards": {
            "test": test_leaderboard,
            "rolling": rolling_leaderboard,
            "selection_basis": selection_basis,
        },
        "prediction_series": prediction_series,
        "rolling": {
            "fold_metrics": fold_metrics.to_dict(orient="records"),
            "predictions": rolling_predictions.to_dict(orient="records"),
            "market_focus_days": market_focus_days.head(12).to_dict(orient="records"),
        },
        "risk_regime_performance": read_optional_table(ANALYSIS_TABLES / "high_volatility_model_performance.csv"),
        "industry_profiles": INDUSTRY_PROFILES,
        "figures": build_figures(),
        "source_files": {
            "export_summary": rel(EXPORT_ROOT / "EXPORT_SUMMARY.json"),
            "test_leaderboard": rel(EXPORT_ROOT / "tables" / "final_test_leaderboard.csv"),
            "rolling_leaderboard": rel(
                ANALYSIS_TABLES / "robustness_comparison_table.csv"
                if (ANALYSIS_TABLES / "robustness_comparison_table.csv").exists()
                else EXPORT_ROOT / "tables" / "rolling_leaderboard.csv"
            ),
            "risk_regime_performance": rel(ANALYSIS_TABLES / "high_volatility_model_performance.csv"),
            "evaluation_figures": rel(ANALYSIS_FIGURES),
            "mainline_predictions": rel(PREDICTION_PATHS["TimeMixer-Fusion"]),
            "structured_daily": rel(STRUCTURED_DAILY),
        },
    }


def write_outputs(payload: dict) -> None:
    sanitized = sanitize_json(payload)
    serialized = json.dumps(sanitized, ensure_ascii=False, indent=2, allow_nan=False)
    DASHBOARD_JSON_PATH.write_text(serialized + "\n", encoding="utf-8")
    DASHBOARD_JS_PATH.write_text(
        "window.DECISION_DASHBOARD_DATA = " + serialized + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(f"Wrote {DASHBOARD_JSON_PATH}")
    print(f"Wrote {DASHBOARD_JS_PATH}")
    print(
        "Mainline:",
        payload["model"]["name"],
        payload["model"]["input_variant"],
        payload["model"]["fusion_method"],
        payload["model_forecast"]["risk_level"],
    )


if __name__ == "__main__":
    main()
