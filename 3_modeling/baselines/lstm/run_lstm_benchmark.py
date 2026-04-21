from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MODELING_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import aggregate_metric_dicts, build_diagnostics, compute_zero_baseline, metric_dict
from common.paths import official_model_dir, time_series_root
from common.reporting import clean_directory, copy_best_run_visuals, save_json, save_prediction_artifacts, save_seed_metric_plot
from common.window_data import (
    build_reference_lookup,
    build_rolling_plan,
    combine_splits,
    compute_future_window,
    infer_input_dim,
    load_latest_variant_window,
    load_split_arrays,
    with_reference,
)
from project_shared.frequency import normalize_frequency


SEED_LIST = [42, 43, 44]


def resolve_training_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :]).squeeze(-1)
        return torch.nn.functional.softplus(pred)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(bundle: dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(bundle["features"], dtype=torch.float32),
        torch.tensor(bundle["labels"], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(model: nn.Module, bundle: dict, device: torch.device) -> tuple[dict, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in make_loader(bundle, batch_size=len(bundle["labels"]), shuffle=False):
            batch_x = batch_x.to(device, non_blocking=True)
            pred = model(batch_x).detach().cpu().numpy()
            preds.append(pred)
            trues.append(batch_y.numpy())
    pred_array = np.maximum(np.concatenate(preds, axis=0), 0.0).astype(np.float32)
    true_array = np.concatenate(trues, axis=0).astype(np.float32)
    return metric_dict(pred_array, true_array, bundle["reference"]), pred_array


def fit_single_run(
    train_bundle: dict,
    valid_bundle: dict,
    seed: int,
    input_dim: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    grad_clip: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> tuple[nn.Module, dict, dict, np.ndarray]:
    set_seed(seed)
    model = LSTMRegressor(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion: nn.Module = nn.SmoothL1Loss(beta=0.02) if loss_name == "huber" else nn.MSELoss()

    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_val_metrics = None
    best_val_pred = None

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in make_loader(train_bundle, batch_size=batch_size, shuffle=True):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        current_val_metrics, current_val_pred = evaluate_model(model, valid_bundle, device)
        if current_val_metrics["rmse"] < best_val_rmse - 1e-12:
            best_val_rmse = current_val_metrics["rmse"]
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            best_val_metrics = current_val_metrics
            best_val_pred = current_val_pred
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch}, best_val_metrics, best_val_pred


def run_lstm_benchmark(
    window_length: int = 12,
    results_dir: Path | None = None,
    root_path: Path | None = None,
    input_variant: str = "fusion",
    fusion_method: str = "late.gru_gate",
    frequency: str = "monthly",
) -> Path:
    frequency = normalize_frequency(frequency)
    official_dir = results_dir or official_model_dir("lstm", window_length, frequency)
    clean_directory(official_dir)

    resolved_root = root_path or time_series_root(input_variant, frequency)
    split_arrays = with_reference(
        load_split_arrays(window_length, root_path=resolved_root, frequency=frequency),
        build_reference_lookup(frequency),
        frequency=frequency,
    )
    combined = combine_splits(split_arrays)
    combined["reference"] = np.concatenate(
        [split_arrays["train"]["reference"], split_arrays["valid"]["reference"], split_arrays["test"]["reference"]],
        axis=0,
    )
    rolling_plan = build_rolling_plan(combined, frequency=frequency)
    final_plan = rolling_plan["final_plan"]

    input_dim = infer_input_dim(window_length, root_path=resolved_root, frequency=frequency)
    device = resolve_training_device()

    batch_size = 16
    max_epochs = 60
    patience = 10
    learning_rate = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0
    loss_name = "huber"
    hidden_size = 64
    num_layers = 1
    dropout = 0.1

    seed_rows = []
    seed_metric_rolling = []
    seed_metric_final_valid = []
    seed_metric_test = []
    best_seed_payload = None

    for seed in SEED_LIST:
        fold_records = []
        for fold in rolling_plan["validation_folds"]:
            _, state_info, valid_metrics, _ = fit_single_run(
                train_bundle=fold["train"],
                valid_bundle=fold["valid"],
                seed=seed + int(fold["fold_index"]),
                input_dim=input_dim,
                device=device,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_name=loss_name,
                grad_clip=grad_clip,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            fold_records.append({**valid_metrics, "best_epoch": state_info["best_epoch"]})

        final_model, state_info, final_valid_metrics, _ = fit_single_run(
            train_bundle=final_plan["train"],
            valid_bundle=final_plan["valid"],
            seed=seed,
            input_dim=input_dim,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_name=loss_name,
            grad_clip=grad_clip,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        train_metrics, _ = evaluate_model(final_model, final_plan["train"], device)
        test_metrics, test_pred = evaluate_model(final_model, final_plan["test"], device)

        row = {
            "seed": seed,
            "rolling_val_rmse_mean": float(np.mean([record["rmse"] for record in fold_records])),
            "rolling_val_rmse_std": float(np.std([record["rmse"] for record in fold_records])),
            "rolling_val_mae_mean": float(np.mean([record["mae"] for record in fold_records])),
            "rolling_val_mae_std": float(np.std([record["mae"] for record in fold_records])),
            "rolling_val_mape_mean": float(np.mean([record["mape"] for record in fold_records])),
            "rolling_val_mse_mean": float(np.mean([record["mse"] for record in fold_records])),
            "rolling_val_direction_acc_mean": float(np.mean([record["direction_acc"] for record in fold_records])),
            "final_valid_rmse": float(final_valid_metrics["rmse"]),
            "final_valid_mae": float(final_valid_metrics["mae"]),
            "final_valid_mape": float(final_valid_metrics["mape"]),
            "final_valid_mse": float(final_valid_metrics["mse"]),
            "final_valid_direction_acc": float(final_valid_metrics["direction_acc"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_mape": float(test_metrics["mape"]),
            "test_mse": float(test_metrics["mse"]),
            "test_direction_acc": float(test_metrics["direction_acc"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_mape": float(train_metrics["mape"]),
            "train_mse": float(train_metrics["mse"]),
            "train_direction_acc": float(train_metrics["direction_acc"]),
            "best_epoch_final": int(state_info["best_epoch"]),
        }
        seed_rows.append(row)
        seed_metric_rolling.append(
            {
                "rmse": row["rolling_val_rmse_mean"],
                "mae": row["rolling_val_mae_mean"],
                "mape": row["rolling_val_mape_mean"],
                "mse": row["rolling_val_mse_mean"],
                "direction_acc": row["rolling_val_direction_acc_mean"],
            }
        )
        seed_metric_final_valid.append(final_valid_metrics)
        seed_metric_test.append(test_metrics)

        candidate_payload = {
            "summary": row,
            "state_dict": deepcopy(final_model.state_dict()),
            "test_pred": test_pred.copy(),
        }
        if best_seed_payload is None or (
            row["final_valid_rmse"],
            row["test_rmse"],
            row["seed"],
        ) < (
            best_seed_payload["summary"]["final_valid_rmse"],
            best_seed_payload["summary"]["test_rmse"],
            best_seed_payload["summary"]["seed"],
        ):
            best_seed_payload = candidate_payload

    seed_df = pd.DataFrame(seed_rows).sort_values(["final_valid_rmse", "test_rmse", "seed"]).reset_index(drop=True)
    seed_df.to_csv(official_dir / "rolling_seed_summary.csv", index=False, encoding="utf-8")
    save_seed_metric_plot(official_dir, seed_df, f"LSTM-{frequency}")

    best_run_dir = official_dir / "best_run"
    best_run_dir.mkdir(parents=True, exist_ok=True)
    best_model = LSTMRegressor(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    best_model.load_state_dict(best_seed_payload["state_dict"])
    torch.save(best_model.state_dict(), best_run_dir / "checkpoint.pth")
    save_prediction_artifacts(
        best_run_dir,
        final_plan["test"]["index"],
        final_plan["test"]["labels"],
        best_seed_payload["test_pred"],
        "LSTM",
        frequency=frequency,
    )
    copy_best_run_visuals(best_run_dir, official_dir)

    future_window, source_index = load_latest_variant_window(window_length, input_variant=input_variant, frequency=frequency)
    best_model.eval()
    with torch.no_grad():
        future_pred = float(best_model(torch.tensor(future_window[None, :, :], dtype=torch.float32, device=device)).cpu().numpy()[0])
    forecast_window = compute_future_window(str(source_index[-1]), frequency=frequency)
    future_payload = {
        "window_length": window_length,
        "input_variant": input_variant,
        "input_window_start": str(source_index[-window_length]),
        "input_window_end": str(source_index[-1]),
        "predicted_value": max(future_pred, 0.0),
        "model": "LSTM-window",
        "fusion_method": fusion_method,
        "frequency": frequency,
        **forecast_window,
    }
    save_json(best_run_dir / "future_forecast.json", future_payload)
    save_json(official_dir / "future_forecast.json", future_payload)

    zero_validation_metrics = [compute_zero_baseline(fold["valid"]) for fold in rolling_plan["validation_folds"]]
    zero_baseline = {
        "rolling_validation": aggregate_metric_dicts(zero_validation_metrics),
        "final_valid": compute_zero_baseline(final_plan["valid"]),
        "final_test": compute_zero_baseline(final_plan["test"]),
    }
    save_json(official_dir / "baseline_zero.json", zero_baseline)

    aggregate_rolling = aggregate_metric_dicts(seed_metric_rolling)
    aggregate_final_valid = aggregate_metric_dicts(seed_metric_final_valid)
    aggregate_test = aggregate_metric_dicts(seed_metric_test)
    diagnostics = build_diagnostics(
        y_pred=best_seed_payload["test_pred"],
        model_val_rmse=best_seed_payload["summary"]["final_valid_rmse"],
        model_test_rmse=best_seed_payload["summary"]["test_rmse"],
        zero_baseline_valid_rmse=zero_baseline["final_valid"]["rmse"],
        zero_baseline_test_rmse=zero_baseline["final_test"]["rmse"],
        aggregate_val_rmse_mean=aggregate_final_valid["rmse_mean"],
        aggregate_test_rmse_mean=aggregate_test["rmse_mean"],
    )
    save_json(official_dir / "diagnostics.json", diagnostics)

    official_metrics = {
        "model": "LSTM-window",
        "fusion_method": fusion_method,
        "input_variant": input_variant,
        "window_length": window_length,
        "input_dim": input_dim,
        "frequency": frequency,
        "target": "target_brent_avg_next_7d",
        "device": str(device),
        "time_series_root": str(Path(resolved_root).resolve()),
        "split_strategy": "rolling_origin",
        "seed_list": SEED_LIST,
        "selection_metric": "final_valid_rmse_mean",
        "aggregate_metrics": {
            "rolling_validation": aggregate_rolling,
            "final_valid": aggregate_final_valid,
            "test": aggregate_test,
        },
        "deployed_run": {
            "seed": int(best_seed_payload["summary"]["seed"]),
            "rolling_val_rmse_mean": float(best_seed_payload["summary"]["rolling_val_rmse_mean"]),
            "final_valid_rmse": float(best_seed_payload["summary"]["final_valid_rmse"]),
            "final_valid_mae": float(best_seed_payload["summary"]["final_valid_mae"]),
            "final_valid_mape": float(best_seed_payload["summary"]["final_valid_mape"]),
            "test_rmse": float(best_seed_payload["summary"]["test_rmse"]),
            "test_mae": float(best_seed_payload["summary"]["test_mae"]),
            "test_mape": float(best_seed_payload["summary"]["test_mape"]),
            "best_epoch_final": int(best_seed_payload["summary"]["best_epoch_final"]),
        },
        "baseline_zero": zero_baseline,
        "diagnostics_status": diagnostics["status"],
        "config": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "loss": loss_name,
            "grad_clip": grad_clip,
        },
    }
    save_json(official_dir / "official_metrics.json", official_metrics)
    pd.DataFrame(
        [
            {
                "model": "LSTM-window",
                "frequency": frequency,
                "window_length": window_length,
                "selection_metric": "final_valid_rmse_mean",
                "final_valid_rmse_mean": aggregate_final_valid["rmse_mean"],
                "final_valid_rmse_std": aggregate_final_valid["rmse_std"],
                "test_rmse_mean": aggregate_test["rmse_mean"],
                "test_rmse_std": aggregate_test["rmse_std"],
                "deployed_final_valid_rmse": best_seed_payload["summary"]["final_valid_rmse"],
                "deployed_test_rmse": best_seed_payload["summary"]["test_rmse"],
                "diagnostics_status": diagnostics["status"],
            }
        ]
    ).to_csv(official_dir / "official_metrics.csv", index=False, encoding="utf-8")
    return official_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official dual-frequency LSTM benchmark.")
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--input-variant", type=str, default="fusion")
    parser.add_argument("--fusion-method", type=str, default="late.gru_gate")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    root_path = Path(args.root_path) if args.root_path else None
    output_dir = run_lstm_benchmark(
        window_length=args.window_length,
        results_dir=results_dir,
        root_path=root_path,
        input_variant=args.input_variant,
        fusion_method=args.fusion_method,
        frequency=args.frequency,
    )
    print(f"Official LSTM output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
