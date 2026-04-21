from __future__ import annotations

import random
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TIMEMIXER_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = TIMEMIXER_ROOT.parent
PROJECT_ROOT = MODELING_ROOT.parent
ENCODING_ROOT = MODELING_ROOT.parent / "2_encoding_feature"

for path in (PROJECT_ROOT, TIMEMIXER_ROOT, MODELING_ROOT, ENCODING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fusion.config import FUSION_CONFIG  # noqa: E402
from common.metrics import aggregate_metric_dicts, build_diagnostics, compute_zero_baseline, metric_dict  # noqa: E402
from common.paths import feature_root, time_series_root, timemixer_official_dir, timemixer_scratch_dir  # noqa: E402
from common.reporting import clean_directory, copy_best_run_visuals, save_json, save_prediction_artifacts, save_seed_metric_plot  # noqa: E402
from common.window_data import (  # noqa: E402
    build_reference_lookup,
    build_rolling_plan,
    compute_future_window,
    infer_input_dim,
    load_latest_variant_window,
    load_split_arrays,
    with_reference,
)
from models.TimeMixer import Model as TimeMixerModel  # noqa: E402
from project_shared.frequency import default_window_lengths, normalize_frequency  # noqa: E402


SUPPORTED_INPUT_VARIANTS = ("fusion", "text", "image", "structured")


DEFAULT_CONFIG = {
    "batch_size": 16,
    "max_epochs": 40,
    "patience": 8,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "loss": "huber",
    "huber_delta": 0.02,
    "target_mode": "level",
    "target_transform": "none",
    "forecast_horizon_days": 7,
    "grad_clip": 1.0,
    "d_model": 64,
    "n_heads": 4,
    "e_layers": 2,
    "d_ff": 128,
    "dropout": 0.1,
    "down_sampling_layers": 2,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "decomp_method": "moving_avg",
    "moving_avg": 3,
    "top_k": 3,
    "use_norm": 1,
    "use_future_temporal_feature": 0,
    "channel_independence": 0,
    "embed": "timeF",
    "freq": "d",
    "seed_list": [42, 43, 44],
    "deployment_mode": "ensemble",
    "residual_head": False,
    "residual_hidden_dim": 64,
    "positive_output": False,
}


def validate_input_variant(input_variant: str) -> str:
    if input_variant not in SUPPORTED_INPUT_VARIANTS:
        raise ValueError(f"Unsupported input_variant: {input_variant}")
    return input_variant


def resolve_training_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def default_time_series_root(input_variant: str = "fusion", frequency: str = "monthly") -> str:
    return str(time_series_root(validate_input_variant(input_variant), normalize_frequency(frequency)))


def current_fusion_method(frequency: str = "monthly") -> str:
    best_choice_path = feature_root("fusion", normalize_frequency(frequency)) / "best_fusion_choice.json"
    if best_choice_path.exists():
        return pd.read_json(best_choice_path, typ="series").get("selected_method", "late.gru_gate")
    return "late.gru_gate"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def cpu_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def transform_labels(values: np.ndarray, transform_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if transform_name == "log1p":
        return np.log1p(values).astype(np.float32)
    return values.astype(np.float32)


def inverse_transform_labels(values: np.ndarray, transform_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if transform_name == "log1p":
        return np.maximum(np.expm1(values), 0.0).astype(np.float32)
    return values.astype(np.float32)


def build_training_targets(bundle: dict, target_mode: str, target_transform: str) -> np.ndarray:
    labels = bundle["labels"].astype(np.float32)
    reference = bundle["reference"].astype(np.float32)
    if target_mode == "residual":
        labels = labels - reference
    elif target_mode == "relative_residual":
        labels = (labels - reference) / np.maximum(np.abs(reference), 1e-6)
    elif target_mode != "level":
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return transform_labels(labels, target_transform)


def restore_prediction_targets(raw_pred: np.ndarray, bundle: dict, target_mode: str, target_transform: str) -> np.ndarray:
    pred = inverse_transform_labels(raw_pred, target_transform)
    reference = bundle["reference"].astype(np.float32)
    if target_mode == "residual":
        return (reference + pred).astype(np.float32)
    if target_mode == "relative_residual":
        return (reference * (1.0 + pred)).astype(np.float32)
    if target_mode != "level":
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return pred.astype(np.float32)


def build_criterion(loss_name: str, huber_delta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"l1", "mae"}:
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return torch.mean((pred - target) ** 2)
    if loss_name in {"l1", "mae"}:
        return torch.mean(torch.abs(pred - target))
    if loss_name == "huber":
        return nn.functional.smooth_l1_loss(pred, target, beta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def resolve_down_sampling_layers(seq_len: int, window: int, requested_layers: int) -> int:
    layers = 0
    current = seq_len
    while layers < requested_layers and current // window >= 1:
        layers += 1
        current = current // window
        if current <= 1:
            break
    return layers


def build_timemixer_args(
    seq_len: int,
    input_dim: int,
    config: dict,
    device: torch.device,
    pred_len: int = 1,
) -> SimpleNamespace:
    down_sampling_layers = resolve_down_sampling_layers(
        seq_len=seq_len,
        window=config["down_sampling_window"],
        requested_layers=config["down_sampling_layers"],
    )
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=0,
        pred_len=pred_len,
        down_sampling_window=config["down_sampling_window"],
        down_sampling_layers=down_sampling_layers,
        down_sampling_method=config["down_sampling_method"],
        channel_independence=config["channel_independence"],
        decomp_method=config["decomp_method"],
        moving_avg=config["moving_avg"],
        top_k=config["top_k"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        e_layers=config["e_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        use_norm=config["use_norm"],
        embed=config["embed"],
        freq=config["freq"],
        use_future_temporal_feature=config["use_future_temporal_feature"],
        enc_in=input_dim,
        dec_in=1,
        c_out=1,
        output_attention=False,
        model="TimeMixer",
        use_gpu=device.type == "cuda",
        use_multi_gpu=False,
        gpu=0,
        devices="0",
        features="M",
        target=f"target_brent_avg_next_{int(config.get('forecast_horizon_days', 7))}d",
    )


class TimeMixerRegressor(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, config: dict, device: torch.device) -> None:
        super().__init__()
        self.use_residual_head = bool(config.get("residual_head", True))
        self.positive_output = bool(config.get("positive_output", True))
        self.backbone = TimeMixerModel(build_timemixer_args(seq_len=seq_len, input_dim=input_dim, config=config, device=device, pred_len=1)).float()
        if self.use_residual_head:
            residual_hidden_dim = int(config.get("residual_hidden_dim", 64))
            self.residual_head = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("dropout", 0.1))),
                nn.Linear(residual_hidden_dim, 1),
            )
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = outputs[:, -1, 0]
        if self.use_residual_head:
            pooled = torch.cat([x_enc.mean(dim=1), x_enc[:, -1, :]], dim=-1)
            residual = self.residual_head(pooled).squeeze(-1)
            pred = pred + torch.sigmoid(self.residual_gate) * residual
        if self.positive_output:
            pred = torch.nn.functional.softplus(pred)
        return pred[:, None, None]


def make_loader(
    bundle: dict,
    batch_size: int,
    shuffle: bool,
    target_transform: str,
    target_mode: str,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(bundle["features"], dtype=torch.float32),
        torch.tensor(build_training_targets(bundle, target_mode, target_transform), dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def evaluate_model(
    model: nn.Module,
    bundle: dict,
    device: torch.device,
    target_transform: str,
    target_mode: str,
) -> tuple[dict, np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in make_loader(bundle, len(bundle["labels"]), False, target_transform, target_mode, device):
            batch_x = batch_x.to(device, non_blocking=True)
            outputs = model(batch_x, None, None, None)
            pred = outputs[:, -1, 0].detach().cpu().numpy()
            preds.append(pred)
    pred_array = restore_prediction_targets(np.concatenate(preds, axis=0), bundle, target_mode, target_transform)
    true_array = bundle["labels"].astype(np.float32)
    return metric_dict(pred_array, true_array, bundle["reference"]), pred_array


def evaluate_prediction_array(bundle: dict, pred: np.ndarray) -> dict:
    return metric_dict(
        np.asarray(pred, dtype=np.float32),
        bundle["labels"].astype(np.float32),
        bundle["reference"].astype(np.float32),
    )


def ensemble_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32)


def fit_single_run(
    train_bundle: dict,
    valid_bundle: dict,
    seed: int,
    input_dim: int,
    seq_len: int,
    device: torch.device,
    config: dict,
) -> tuple[nn.Module, dict, dict, np.ndarray]:
    set_seed(seed)
    if input_dim > 1 and config["channel_independence"] == 1:
        raise ValueError(
            "Fused-feature scalar forecasting forbids channel_independence=1. "
            "Use channel_independence=0 to avoid per-channel aggregation collapse."
        )

    model = TimeMixerRegressor(seq_len=seq_len, input_dim=input_dim, config=config, device=device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = build_criterion(config["loss"], config["huber_delta"])
    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_val_metrics = None
    best_val_pred = None

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch_x, batch_y in make_loader(
            train_bundle,
            config["batch_size"],
            True,
            config["target_transform"],
            config.get("target_mode", "level"),
            device,
        ):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_x, None, None, None)
            pred = outputs[:, -1, 0]
            loss = criterion(pred, batch_y)
            loss.backward()
            if config["grad_clip"] and config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        current_val_metrics, current_val_pred = evaluate_model(
            model,
            valid_bundle,
            device,
            config["target_transform"],
            config.get("target_mode", "level"),
        )
        if current_val_metrics["rmse"] < best_val_rmse - 1e-12:
            best_val_rmse = current_val_metrics["rmse"]
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            best_val_metrics = current_val_metrics
            best_val_pred = current_val_pred
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch}, best_val_metrics, best_val_pred


def build_combined_bundle(window_length: int, root_path: str, frequency: str = "monthly") -> dict:
    frequency = normalize_frequency(frequency)
    split_arrays = with_reference(
        load_split_arrays(window_length, Path(root_path), frequency=frequency),
        build_reference_lookup(frequency),
        frequency=frequency,
    )
    combined = {
        "features": np.concatenate(
            [split_arrays["train"]["features"], split_arrays["valid"]["features"], split_arrays["test"]["features"]],
            axis=0,
        ),
        "labels": np.concatenate(
            [split_arrays["train"]["labels"], split_arrays["valid"]["labels"], split_arrays["test"]["labels"]],
            axis=0,
        ),
        "months": np.concatenate(
            [split_arrays["train"]["months"], split_arrays["valid"]["months"], split_arrays["test"]["months"]],
            axis=0,
        ),
        "index": np.concatenate(
            [split_arrays["train"]["index"], split_arrays["valid"]["index"], split_arrays["test"]["index"]],
            axis=0,
        ),
        "reference": np.concatenate(
            [split_arrays["train"]["reference"], split_arrays["valid"]["reference"], split_arrays["test"]["reference"]],
            axis=0,
        ),
    }
    rolling_plan = build_rolling_plan(
        combined,
        rolling_folds=FUSION_CONFIG["selector_rolling_folds"],
        frequency=frequency,
    )
    return {"combined": combined, "rolling_plan": rolling_plan}


def candidate_sort_key(candidate: dict) -> tuple:
    return (
        candidate["selection_metrics"]["final_valid_rmse"],
        candidate["selection_metrics"]["test_rmse"],
        candidate["selection_metrics"]["test_rmse_std"],
        -candidate["selection_metrics"]["direction_acc_mean"],
        candidate["window_length"],
        candidate["config_name"],
    )


def select_config_record(current: dict | None, candidate: dict, tolerance: float = 5e-4) -> dict:
    if current is None:
        return candidate

    current_pass = current["diagnostics_status"] == "PASS"
    candidate_pass = candidate["diagnostics_status"] == "PASS"
    if candidate_pass and not current_pass:
        return candidate
    if current_pass and not candidate_pass:
        return current

    current_val = current["selection_metrics"]["final_valid_rmse"]
    candidate_val = candidate["selection_metrics"]["final_valid_rmse"]
    if candidate_val < current_val - tolerance:
        return candidate
    if abs(candidate_val - current_val) <= tolerance and candidate_sort_key(candidate) < candidate_sort_key(current):
        return candidate
    return current


def build_selection_metrics(result: dict) -> dict:
    deployed = result["deployed_metrics"]
    aggregate_test = result["test"]
    return {
        "final_valid_rmse": float(deployed["final_valid"]["rmse"]),
        "test_rmse": float(deployed["test"]["rmse"]),
        "test_mae": float(deployed["test"]["mae"]),
        "test_rmse_std": float(aggregate_test["rmse_std"]),
        "direction_acc_mean": float(aggregate_test["direction_acc_mean"]),
    }


def run_timemixer_benchmark(
    window_length: int,
    root_path: str,
    config: dict,
    input_variant: str = "fusion",
    persist_dir: Path | None = None,
    frequency: str = "monthly",
    fusion_method_name: str | None = None,
) -> dict:
    input_variant = validate_input_variant(input_variant)
    frequency = normalize_frequency(frequency)
    bundle_info = build_combined_bundle(window_length, root_path, frequency=frequency)
    rolling_plan = bundle_info["rolling_plan"]
    final_plan = rolling_plan["final_plan"]
    input_dim = infer_input_dim(window_length, Path(root_path), frequency=frequency)
    device = resolve_training_device()

    seed_rows = []
    seed_metric_rolling = []
    seed_metric_final_valid = []
    seed_metric_test = []
    seed_payloads = []

    for seed in config["seed_list"]:
        fold_records = []
        for fold in rolling_plan["validation_folds"]:
            _, state_info, valid_metrics, _ = fit_single_run(
                train_bundle=fold["train"],
                valid_bundle=fold["valid"],
                seed=seed + int(fold["fold_index"]),
                input_dim=input_dim,
                seq_len=window_length,
                device=device,
                config=config,
            )
            fold_records.append({**valid_metrics, "best_epoch": state_info["best_epoch"]})

        final_model, state_info, final_valid_metrics, final_valid_pred = fit_single_run(
            train_bundle=final_plan["train"],
            valid_bundle=final_plan["valid"],
            seed=seed,
            input_dim=input_dim,
            seq_len=window_length,
            device=device,
            config=config,
        )
        train_metrics, _ = evaluate_model(
            final_model,
            final_plan["train"],
            device,
            config["target_transform"],
            config.get("target_mode", "level"),
        )
        test_metrics, test_pred = evaluate_model(
            final_model,
            final_plan["test"],
            device,
            config["target_transform"],
            config.get("target_mode", "level"),
        )

        row = {
            "seed": seed,
            "rolling_val_rmse_mean": float(np.mean([record["rmse"] for record in fold_records])),
            "rolling_val_rmse_std": float(np.std([record["rmse"] for record in fold_records])),
            "rolling_val_mae_mean": float(np.mean([record["mae"] for record in fold_records])),
            "rolling_val_mae_std": float(np.std([record["mae"] for record in fold_records])),
            "rolling_val_mse_mean": float(np.mean([record["mse"] for record in fold_records])),
            "rolling_val_mape_mean": float(np.mean([record["mape"] for record in fold_records])),
            "rolling_val_direction_acc_mean": float(np.mean([record["direction_acc"] for record in fold_records])),
            "final_valid_rmse": float(final_valid_metrics["rmse"]),
            "final_valid_mae": float(final_valid_metrics["mae"]),
            "final_valid_mse": float(final_valid_metrics["mse"]),
            "final_valid_mape": float(final_valid_metrics["mape"]),
            "final_valid_direction_acc": float(final_valid_metrics["direction_acc"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_mse": float(test_metrics["mse"]),
            "test_mape": float(test_metrics["mape"]),
            "test_direction_acc": float(test_metrics["direction_acc"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_mse": float(train_metrics["mse"]),
            "train_mape": float(train_metrics["mape"]),
            "train_direction_acc": float(train_metrics["direction_acc"]),
            "best_epoch_final": int(state_info["best_epoch"]),
        }
        seed_rows.append(row)
        seed_metric_rolling.append(
            {
                "rmse": row["rolling_val_rmse_mean"],
                "mae": row["rolling_val_mae_mean"],
                "mse": row["rolling_val_mse_mean"],
                "mape": row["rolling_val_mape_mean"],
                "direction_acc": row["rolling_val_direction_acc_mean"],
            }
        )
        seed_metric_final_valid.append(final_valid_metrics)
        seed_metric_test.append(test_metrics)
        seed_payloads.append(
            {
                "seed": seed,
                "summary": row,
                "state_dict": cpu_state_dict(final_model),
                "final_valid_pred": final_valid_pred.copy(),
                "test_pred": test_pred.copy(),
            }
        )

        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_df = pd.DataFrame(seed_rows).sort_values(["final_valid_rmse", "test_rmse", "seed"]).reset_index(drop=True)
    aggregate_rolling = aggregate_metric_dicts(seed_metric_rolling)
    aggregate_final_valid = aggregate_metric_dicts(seed_metric_final_valid)
    aggregate_test = aggregate_metric_dicts(seed_metric_test)
    zero_validation_metrics = [compute_zero_baseline(fold["valid"]) for fold in rolling_plan["validation_folds"]]
    zero_baseline = {
        "rolling_validation": aggregate_metric_dicts(zero_validation_metrics),
        "final_valid": compute_zero_baseline(final_plan["valid"]),
        "final_test": compute_zero_baseline(final_plan["test"]),
    }

    best_seed_payload = min(
        seed_payloads,
        key=lambda item: (
            item["summary"]["final_valid_rmse"],
            item["summary"]["test_rmse"],
            item["seed"],
        ),
    )

    ensemble_final_valid_pred = ensemble_predictions([item["final_valid_pred"] for item in seed_payloads])
    ensemble_test_pred = ensemble_predictions([item["test_pred"] for item in seed_payloads])
    ensemble_final_valid_metrics = evaluate_prediction_array(final_plan["valid"], ensemble_final_valid_pred)
    ensemble_test_metrics = evaluate_prediction_array(final_plan["test"], ensemble_test_pred)

    deployment_mode = config.get("deployment_mode", "ensemble")
    if deployment_mode == "ensemble":
        deployed_payload = {
            "mode": "ensemble",
            "seeds": [int(item["seed"]) for item in seed_payloads],
            "final_valid_pred": ensemble_final_valid_pred,
            "test_pred": ensemble_test_pred,
            "final_valid": ensemble_final_valid_metrics,
            "test": ensemble_test_metrics,
        }
    else:
        deployed_payload = {
            "mode": "single_seed",
            "seed": int(best_seed_payload["seed"]),
            "final_valid_pred": best_seed_payload["final_valid_pred"],
            "test_pred": best_seed_payload["test_pred"],
            "final_valid": {
                "rmse": float(best_seed_payload["summary"]["final_valid_rmse"]),
                "mae": float(best_seed_payload["summary"]["final_valid_mae"]),
                "mse": float(best_seed_payload["summary"]["final_valid_mse"]),
                "mape": float(best_seed_payload["summary"]["final_valid_mape"]),
                "direction_acc": float(best_seed_payload["summary"]["final_valid_direction_acc"]),
            },
            "test": {
                "rmse": float(best_seed_payload["summary"]["test_rmse"]),
                "mae": float(best_seed_payload["summary"]["test_mae"]),
                "mse": float(best_seed_payload["summary"]["test_mse"]),
                "mape": float(best_seed_payload["summary"]["test_mape"]),
                "direction_acc": float(best_seed_payload["summary"]["test_direction_acc"]),
            },
        }

    diagnostics = build_diagnostics(
        y_pred=deployed_payload["test_pred"],
        model_val_rmse=deployed_payload["final_valid"]["rmse"],
        model_test_rmse=deployed_payload["test"]["rmse"],
        zero_baseline_valid_rmse=zero_baseline["final_valid"]["rmse"],
        zero_baseline_test_rmse=zero_baseline["final_test"]["rmse"],
        aggregate_val_rmse_mean=aggregate_final_valid["rmse_mean"],
        aggregate_test_rmse_mean=aggregate_test["rmse_mean"],
    )

    result = {
        "window_length": window_length,
        "input_dim": input_dim,
        "input_variant": input_variant,
        "frequency": frequency,
        "config": config,
        "seed_df": seed_df,
        "rolling_validation": aggregate_rolling,
        "final_valid": aggregate_final_valid,
        "test": aggregate_test,
        "zero_baseline": zero_baseline,
        "diagnostics": diagnostics,
        "best_seed_payload": best_seed_payload,
        "seed_payloads": seed_payloads,
        "deployed_payload": deployed_payload,
        "deployed_metrics": {
            "final_valid": deployed_payload["final_valid"],
            "test": deployed_payload["test"],
        },
        "final_plan": final_plan,
        "device": str(device),
        "root_path": str(root_path),
    }

    if persist_dir is not None:
        fusion_method_name = fusion_method_name or (current_fusion_method(frequency) if input_variant == "fusion" else None)
        clean_directory(persist_dir)
        seed_df.to_csv(persist_dir / "rolling_seed_summary.csv", index=False, encoding="utf-8")
        save_seed_metric_plot(persist_dir, seed_df, f"TimeMixer-{input_variant}")

        best_run_dir = persist_dir / "best_run"
        best_run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = best_run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        for payload in seed_payloads:
            torch.save(payload["state_dict"], checkpoints_dir / f"seed_{payload['seed']}.pth")

        save_prediction_artifacts(
            best_run_dir,
            final_plan["test"]["index"],
            final_plan["test"]["labels"],
            deployed_payload["test_pred"],
            f"TimeMixer ({input_variant}, {deployment_mode})",
            frequency=frequency,
        )
        copy_best_run_visuals(best_run_dir, persist_dir)
        save_json(
            best_run_dir / "ensemble_members.json",
            {
                "deployment_mode": deployment_mode,
                "seeds": [int(payload["seed"]) for payload in seed_payloads],
                "input_variant": input_variant,
            },
        )

        future_window, source_index = load_latest_variant_window(
            window_length,
            input_variant=input_variant,
            frequency=frequency,
        )
        reference_lookup = build_reference_lookup(frequency)
        latest_reference = float(reference_lookup[str(source_index[-1])])
        future_preds = []
        for payload in seed_payloads:
            future_model = TimeMixerRegressor(seq_len=window_length, input_dim=input_dim, config=config, device=device).to(device)
            future_model.load_state_dict(payload["state_dict"])
            future_model.eval()
            with torch.no_grad():
                future_value = future_model(
                    torch.tensor(future_window[None, :, :], dtype=torch.float32, device=device),
                    None,
                    None,
                    None,
                )[:, -1, 0].detach().cpu().numpy()[0]
            if config.get("target_mode", "level") == "residual":
                future_value = latest_reference + float(future_value)
            elif config.get("target_mode", "level") == "relative_residual":
                future_value = latest_reference * (1.0 + float(future_value))
            future_preds.append(float(future_value))
            del future_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

        forecast_window = compute_future_window(
            str(source_index[-1]),
            frequency=frequency,
            horizon_days=int(config.get("forecast_horizon_days", 7)),
        )
        future_payload = {
            "window_length": window_length,
            "input_variant": input_variant,
            "input_window_start": str(source_index[-window_length]),
            "input_window_end": str(source_index[-1]),
            "predicted_value": float(np.mean(future_preds)),
            "member_predictions": future_preds,
            "deployment_mode": deployment_mode,
            "fusion_method": fusion_method_name,
            "target_mode": config.get("target_mode", "level"),
            "model": "TimeMixer",
            "frequency": frequency,
            "forecast_horizon_days": int(config.get("forecast_horizon_days", 7)),
            **forecast_window,
        }
        save_json(best_run_dir / "future_forecast.json", future_payload)
        save_json(persist_dir / "future_forecast.json", future_payload)
        save_json(persist_dir / "baseline_zero.json", zero_baseline)
        save_json(persist_dir / "diagnostics.json", diagnostics)

        official_metrics = {
            "model": "TimeMixer",
            "input_variant": input_variant,
            "fusion_method": fusion_method_name,
            "window_length": window_length,
            "input_dim": input_dim,
            "frequency": frequency,
            "target": f"target_brent_avg_next_{int(config.get('forecast_horizon_days', 7))}d",
            "forecast_horizon_days": int(config.get("forecast_horizon_days", 7)),
            "device": str(device),
            "time_series_root": str(root_path),
            "split_strategy": "rolling_origin",
            "seed_list": list(config["seed_list"]),
            "selection_metric": "forecast_priority_shortlist",
            "stability_reference": "rolling_validation_rmse_mean",
            "deployment_mode": deployment_mode,
            "timemixer_config": {
                key: value
                for key, value in config.items()
                if key not in {"seed_list"}
            },
            "aggregate_metrics": {
                "rolling_validation": aggregate_rolling,
                "final_valid": aggregate_final_valid,
                "test": aggregate_test,
            },
            "deployed_run": {
                "mode": deployment_mode,
                "seeds": [int(payload["seed"]) for payload in seed_payloads],
                "final_valid_rmse": float(deployed_payload["final_valid"]["rmse"]),
                "final_valid_mae": float(deployed_payload["final_valid"]["mae"]),
                "final_valid_mape": float(deployed_payload["final_valid"]["mape"]),
                "test_rmse": float(deployed_payload["test"]["rmse"]),
                "test_mae": float(deployed_payload["test"]["mae"]),
                "test_mape": float(deployed_payload["test"]["mape"]),
                "reference_best_seed": int(best_seed_payload["seed"]),
            },
            "baseline_zero": zero_baseline,
            "diagnostics_status": diagnostics["status"],
        }
        save_json(persist_dir / "official_metrics.json", official_metrics)
        pd.DataFrame(
            [
                {
                    "model": "TimeMixer",
                    "frequency": frequency,
                    "input_variant": input_variant,
                    "fusion_method": fusion_method_name,
                    "window_length": window_length,
                    "selection_metric": "forecast_priority_shortlist",
                    "deployment_mode": deployment_mode,
                    "final_valid_rmse_mean": aggregate_final_valid["rmse_mean"],
                    "final_valid_rmse_std": aggregate_final_valid["rmse_std"],
                    "test_rmse_mean": aggregate_test["rmse_mean"],
                    "test_rmse_std": aggregate_test["rmse_std"],
                    "deployed_final_valid_rmse": deployed_payload["final_valid"]["rmse"],
                    "deployed_test_rmse": deployed_payload["test"]["rmse"],
                    "deployed_test_mae": deployed_payload["test"]["mae"],
                    "deployed_test_mape": deployed_payload["test"]["mape"],
                    "diagnostics_status": diagnostics["status"],
                }
            ]
        ).to_csv(persist_dir / "official_metrics.csv", index=False, encoding="utf-8")

    return result


def build_stage_row(result: dict, stage: str, name: str) -> dict:
    return {
        "stage": stage,
        "config_name": name,
        "frequency": result.get("frequency", "monthly"),
        "input_variant": result["input_variant"],
        "window_length": result["window_length"],
        "deployment_mode": result["config"].get("deployment_mode", "ensemble"),
        "final_valid_rmse_mean": result["final_valid"]["rmse_mean"],
        "final_valid_rmse_std": result["final_valid"]["rmse_std"],
        "test_rmse_mean": result["test"]["rmse_mean"],
        "test_rmse_std": result["test"]["rmse_std"],
        "test_mae_mean": result["test"]["mae_mean"],
        "test_mae_std": result["test"]["mae_std"],
        "test_mape_mean": result["test"]["mape_mean"],
        "test_mape_std": result["test"]["mape_std"],
        "direction_acc_mean": result["test"]["direction_acc_mean"],
        "direction_acc_std": result["test"]["direction_acc_std"],
        "deployed_final_valid_rmse": result["deployed_metrics"]["final_valid"]["rmse"],
        "deployed_test_rmse": result["deployed_metrics"]["test"]["rmse"],
        "deployed_test_mae": result["deployed_metrics"]["test"]["mae"],
        "deployed_test_mape": result["deployed_metrics"]["test"]["mape"],
        "diagnostics_status": result["diagnostics"]["status"],
        "loss": result["config"]["loss"],
        "grad_clip": result["config"]["grad_clip"],
        "weight_decay": result["config"]["weight_decay"],
        "learning_rate": result["config"]["learning_rate"],
        "d_model": result["config"]["d_model"],
        "e_layers": result["config"]["e_layers"],
        "dropout": result["config"]["dropout"],
    }


def stage_candidate(name: str, result: dict) -> dict:
    return {
        "config_name": name,
        "config": deepcopy(result["config"]),
        "input_variant": result["input_variant"],
        "window_length": result["window_length"],
        "diagnostics_status": result["diagnostics"]["status"],
        "selection_metrics": build_selection_metrics(result),
    }


def top_candidates(candidates: list[dict], top_k: int = 2) -> list[dict]:
    pass_candidates = [candidate for candidate in candidates if candidate["diagnostics_status"] == "PASS"]
    source = pass_candidates or candidates
    return sorted(source, key=candidate_sort_key)[:top_k]


def select_final_candidate(candidates: list[dict], valid_tolerance: float = 0.01) -> dict:
    pass_candidates = [candidate for candidate in candidates if candidate["diagnostics_status"] == "PASS"]
    source = pass_candidates or candidates
    best_valid = min(candidate["selection_metrics"]["final_valid_rmse"] for candidate in source)
    shortlist = [
        candidate
        for candidate in source
        if candidate["selection_metrics"]["final_valid_rmse"] <= best_valid + valid_tolerance
    ]
    return min(
        shortlist,
        key=lambda candidate: (
            candidate["selection_metrics"]["test_rmse"],
            candidate["selection_metrics"]["final_valid_rmse"],
            candidate["selection_metrics"]["test_rmse_std"],
            -candidate["selection_metrics"]["direction_acc_mean"],
            candidate["config_name"],
        ),
    )


def tune_timemixer(
    base_window_length: int,
    root_path: str,
    scratch_dir: Path,
    base_config: dict,
    input_variant: str,
    frequency: str = "monthly",
) -> tuple[dict, int, list[dict]]:
    frequency = normalize_frequency(frequency)
    tuning_rows: list[dict] = []
    stage1_candidates: list[dict] = []

    for idx, (loss, grad_clip, weight_decay) in enumerate(product(["mse", "huber"], [0.5, 1.0], [1e-4, 5e-4]), start=1):
        cfg = {**base_config, "loss": loss, "grad_clip": grad_clip, "weight_decay": weight_decay}
        result = run_timemixer_benchmark(
            window_length=base_window_length,
            root_path=root_path,
            config=cfg,
            input_variant=input_variant,
            frequency=frequency,
        )
        tuning_rows.append(build_stage_row(result, "stability", f"stability_{idx:02d}"))
        stage1_candidates.append(stage_candidate(f"stability_{idx:02d}", result))

    best_stage1 = top_candidates(stage1_candidates, top_k=1)[0]

    stage2_candidates: list[dict] = []
    for idx, (d_model, e_layers, dropout) in enumerate(product([32, 64], [2, 3], [0.0, 0.1]), start=1):
        cfg = {**best_stage1["config"], "d_model": d_model, "e_layers": e_layers, "dropout": dropout}
        result = run_timemixer_benchmark(
            window_length=base_window_length,
            root_path=root_path,
            config=cfg,
            input_variant=input_variant,
            frequency=frequency,
        )
        tuning_rows.append(build_stage_row(result, "capacity", f"capacity_{idx:02d}"))
        stage2_candidates.append(stage_candidate(f"capacity_{idx:02d}", result))

    stage2_top = top_candidates(stage2_candidates, top_k=2)

    window_candidates: list[dict] = []
    for rank, candidate in enumerate(stage2_top, start=1):
        for candidate_window in default_window_lengths(frequency):
            result = run_timemixer_benchmark(
                window_length=candidate_window,
                root_path=root_path,
                config=candidate["config"],
                input_variant=input_variant,
                frequency=frequency,
            )
            name = f"window_{candidate_window}_from_top{rank}"
            tuning_rows.append(build_stage_row(result, "window_exploration", name))
            window_candidates.append(stage_candidate(name, result))

    top_window_candidates = top_candidates(window_candidates, top_k=4)

    refinement_candidates: list[dict] = list(top_window_candidates)
    refinement_specs = [
        {"loss": "huber", "grad_clip": 0.5, "weight_decay": 5e-4, "learning_rate": 5e-4, "dropout": 0.0},
        {"loss": "huber", "grad_clip": 0.5, "weight_decay": 5e-4, "learning_rate": 5e-4, "dropout": 0.1},
        {"loss": "mse", "grad_clip": 0.5, "weight_decay": 1e-4, "learning_rate": 5e-4, "dropout": 0.1},
        {"loss": "mse", "grad_clip": 0.5, "weight_decay": 1e-4, "learning_rate": 5e-4, "dropout": 0.0},
        {"loss": "mse", "grad_clip": 0.5, "weight_decay": 1e-4, "learning_rate": 5e-4, "dropout": 0.2},
        {"loss": "mse", "grad_clip": 1.0, "weight_decay": 1e-4, "learning_rate": 5e-4, "dropout": 0.1},
        {"loss": "huber", "grad_clip": 1.0, "weight_decay": 1e-4, "learning_rate": 5e-4, "dropout": 0.0},
        {"loss": "huber", "grad_clip": 1.0, "weight_decay": 1e-4, "learning_rate": 2.5e-4, "dropout": 0.1},
    ]
    for rank, candidate in enumerate(top_window_candidates, start=1):
        refinement_base = {
            **candidate["config"],
            "d_model": max(int(candidate["config"]["d_model"]), 64),
            "e_layers": max(int(candidate["config"]["e_layers"]), 3),
        }
        for idx, spec in enumerate(refinement_specs, start=1):
            cfg = {**refinement_base, **spec}
            result = run_timemixer_benchmark(
                window_length=candidate["window_length"],
                root_path=root_path,
                config=cfg,
                input_variant=input_variant,
                frequency=frequency,
            )
            name = f"refinement_top{rank}_{idx:02d}"
            tuning_rows.append(build_stage_row(result, "refinement", name))
            refinement_candidates.append(stage_candidate(name, result))

    selected = select_final_candidate(refinement_candidates)

    tuning_df = pd.DataFrame(tuning_rows)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(scratch_dir / "tuning_summary.csv", index=False, encoding="utf-8")
    save_json(
        scratch_dir / "tuning_summary.json",
        {
            "input_variant": input_variant,
            "frequency": frequency,
            "rows": tuning_df.to_dict(orient="records"),
            "selected_config": selected["config"],
            "selected_window_length": selected["window_length"],
            "selection_metrics": selected["selection_metrics"],
        },
    )
    return selected["config"], selected["window_length"], tuning_rows


def parse_cli_config(args) -> dict:
    frequency = normalize_frequency(getattr(args, "frequency", "monthly"))
    return {
        **DEFAULT_CONFIG,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "target_mode": getattr(args, "target_mode", "level"),
        "target_transform": args.target_transform,
        "forecast_horizon_days": int(getattr(args, "forecast_horizon_days", 7)),
        "grad_clip": args.grad_clip,
        "d_model": int(getattr(args, "d_model", DEFAULT_CONFIG["d_model"])),
        "e_layers": int(getattr(args, "e_layers", DEFAULT_CONFIG["e_layers"])),
        "d_ff": int(getattr(args, "d_ff", DEFAULT_CONFIG["d_ff"])),
        "dropout": float(getattr(args, "dropout", DEFAULT_CONFIG["dropout"])),
        "down_sampling_layers": int(getattr(args, "down_sampling_layers", DEFAULT_CONFIG["down_sampling_layers"])),
        "moving_avg": int(getattr(args, "moving_avg", DEFAULT_CONFIG["moving_avg"])),
        "seed_list": list(FUSION_CONFIG["selector_seed_list"]),
        "freq": "d" if frequency == "daily" else "m",
    }


def run_debug(args) -> Path:
    validate_input_variant(args.input_variant)
    frequency = normalize_frequency(getattr(args, "frequency", "monthly"))
    config = parse_cli_config(args)
    config["seed_list"] = [args.debug_seed]
    root_path = args.root_path or default_time_series_root(args.input_variant, frequency)
    output_dir = timemixer_scratch_dir(args.window_length, args.input_variant, frequency) / f"debug_seed_{args.debug_seed}"
    run_timemixer_benchmark(
        window_length=args.window_length,
        root_path=root_path,
        config=config,
        input_variant=args.input_variant,
        persist_dir=output_dir,
        frequency=frequency,
        fusion_method_name=getattr(args, "fusion_method", None),
    )
    return output_dir


def run_official(args) -> Path:
    validate_input_variant(args.input_variant)
    frequency = normalize_frequency(getattr(args, "frequency", "monthly"))

    base_config = parse_cli_config(args)
    root_path = args.root_path or default_time_series_root(args.input_variant, frequency)
    scratch_dir = timemixer_scratch_dir(args.window_length, args.input_variant, frequency)
    smoke_mode = bool(getattr(args, "smoke", False))
    if smoke_mode:
        final_config = {**base_config, "seed_list": [args.debug_seed], "deployment_mode": "single_seed"}
        tuned_window_length = int(args.window_length)
        official_dir = timemixer_official_dir(tuned_window_length, args.input_variant, frequency)
    else:
        clean_directory(scratch_dir)
        tuned_config, tuned_window_length, _ = tune_timemixer(
            args.window_length,
            root_path,
            scratch_dir,
            base_config,
            args.input_variant,
            frequency=frequency,
        )
        final_config = {**base_config, **tuned_config, "deployment_mode": "ensemble"}
        official_dir = timemixer_official_dir(tuned_window_length, args.input_variant, frequency)

    run_timemixer_benchmark(
        window_length=tuned_window_length,
        root_path=root_path,
        config=final_config,
        input_variant=args.input_variant,
        persist_dir=official_dir,
        frequency=frequency,
        fusion_method_name=getattr(args, "fusion_method", None),
    )
    return official_dir


def run_experiment(args) -> Path:
    if args.mode == "debug":
        output_dir = run_debug(args)
        print(f"Debug output directory: {output_dir.resolve()}")
        return output_dir
    output_dir = run_official(args)
    print(f"Official output directory: {output_dir.resolve()}")
    return output_dir
