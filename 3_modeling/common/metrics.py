from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    return float(np.mean(np.abs(pred - true)))


def mape(pred: np.ndarray, true: np.ndarray, epsilon: float = 1e-6) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs(pred - true) / denominator))


def direction_accuracy(pred: np.ndarray, true: np.ndarray, reference: np.ndarray) -> float:
    pred_direction = np.sign(np.asarray(pred) - np.asarray(reference))
    true_direction = np.sign(np.asarray(true) - np.asarray(reference))
    return float(np.mean(pred_direction == true_direction))


def metric_dict(pred: np.ndarray, true: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mae": mae(pred, true),
        "mape": mape(pred, true),
        "direction_acc": direction_accuracy(pred, true, reference),
    }


def aggregate_metric_dicts(metric_dicts: list[dict]) -> dict:
    result = {}
    keys = sorted({key for item in metric_dicts for key in item.keys()})
    for key in keys:
        values = [float(item[key]) for item in metric_dicts if key in item]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def compute_zero_baseline(bundle: dict) -> dict:
    pred = np.asarray(bundle["reference"], dtype=np.float32)
    return metric_dict(pred, bundle["labels"], bundle["reference"])


def build_diagnostics(
    y_pred: np.ndarray,
    model_val_rmse: float,
    model_test_rmse: float,
    zero_baseline_valid_rmse: float,
    zero_baseline_test_rmse: float,
    aggregate_val_rmse_mean: float | None = None,
    aggregate_test_rmse_mean: float | None = None,
) -> dict:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    pred_std = float(np.std(y_pred))
    constant_prediction = bool(pred_std < 1e-8)
    matches_naive_baseline = bool(
        abs(model_val_rmse - zero_baseline_valid_rmse) < 1e-8
        and abs(model_test_rmse - zero_baseline_test_rmse) < 1e-8
    )
    better_than_naive_val = bool(model_val_rmse < zero_baseline_valid_rmse)
    not_worse_than_naive_test = bool(model_test_rmse <= zero_baseline_test_rmse + 1e-8)

    issues = []
    if constant_prediction:
        issues.append("test predictions are nearly constant")
    if matches_naive_baseline:
        issues.append("metrics are effectively identical to the naive reference baseline")
    if not better_than_naive_val:
        issues.append("validation RMSE does not beat the naive reference baseline")
    if not not_worse_than_naive_test:
        issues.append("test RMSE is worse than the naive reference baseline")

    return {
        "status": "PASS" if not issues else "FAILED_DIAGNOSTIC",
        "issues": issues,
        "checks": {
            "prediction_std_gt_threshold": not constant_prediction,
            "metrics_not_equal_naive_baseline": not matches_naive_baseline,
            "validation_beats_naive_baseline": better_than_naive_val,
            "test_not_worse_than_naive_baseline": not_worse_than_naive_test,
        },
        "best_run_prediction_std": pred_std,
        "best_run_prediction_mean": float(np.mean(y_pred)),
        "naive_baseline_val_rmse": float(zero_baseline_valid_rmse),
        "naive_baseline_test_rmse": float(zero_baseline_test_rmse),
        "model_val_rmse": float(model_val_rmse),
        "model_test_rmse": float(model_test_rmse),
        "aggregate_val_rmse_mean": None if aggregate_val_rmse_mean is None else float(aggregate_val_rmse_mean),
        "aggregate_test_rmse_mean": None if aggregate_test_rmse_mean is None else float(aggregate_test_rmse_mean),
    }
