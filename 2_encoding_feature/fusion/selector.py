from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from fusion.config import FUSION_CONFIG, STAGE_PRIORITY, output_config, split_sizes
from fusion.modules import build_fusion_module
from fusion.timemixer_backend import FusionTimeMixerForecaster, TimeMixerSelectorTrainer
from time_series.dataset_builder import DatasetBuilder
from time_series.window_builder import WindowBuilder
from project_shared.frequency import default_window_length, default_window_lengths, normalize_frequency
from project_shared.paths import STRUCTURED_DAILY_PATH, STRUCTURED_MONTHLY_DERIVED_PATH
from project_shared.targets import (
    REFERENCE_PRICE_COLUMN,
    STEP_RETURN_COLUMN,
    STEP_VOLATILITY_COLUMN,
    compute_forward_average,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def direction_accuracy(pred: np.ndarray, true: np.ndarray, reference: np.ndarray) -> float:
    pred_direction = np.sign(pred - reference)
    true_direction = np.sign(true - reference)
    return float(np.mean(pred_direction == true_direction))


class SequencePoolRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(FUSION_CONFIG["dropout"]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(FUSION_CONFIG["dropout"]),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        mean_pool = seq_feat.mean(dim=1)
        last_pool = seq_feat[:, -1, :]
        pooled = torch.cat([mean_pool, last_pool], dim=-1)
        return self.mlp(pooled).squeeze(-1)


class StructuredFusionRegressor(nn.Module):
    def __init__(
        self,
        stage: str,
        method: str,
        structured_dim: int,
        text_dim: int,
        image_dim: int,
        context_dim: int,
        fused_dim: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.fused_dim = fused_dim
        self.fusion_module = build_fusion_module(
            stage,
            method,
            text_dim,
            image_dim,
            fused_dim,
            context_dim=context_dim,
            alpha=alpha,
        )
        self.structured_proj = nn.Linear(structured_dim, fused_dim)
        self.regressor = SequencePoolRegressor(input_dim=fused_dim * 2, hidden_dim=FUSION_CONFIG["selector_hidden_dim"])

    def auxiliary_loss(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.fusion_module, "auxiliary_loss"):
            return self.fusion_module.auxiliary_loss(aux)
        return aux["final_seq"].new_tensor(0.0)

    def forward(
        self,
        structured_feat: torch.Tensor,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor,
        context_feat: torch.Tensor,
        return_aux: bool = False,
    ):
        fusion_out = self.fusion_module(text_feat, image_feat, context_feat=context_feat, return_aux=True)
        fused_text_image = fusion_out["fused_feat"]
        structured_proj = self.structured_proj(structured_feat)
        final_seq = torch.cat([structured_proj, fused_text_image], dim=-1)
        pred = self.regressor(final_seq)

        if not return_aux:
            return pred

        output = {
            "prediction": pred,
            "structured_proj": structured_proj,
            "fused_text_image": fused_text_image,
            "final_seq": final_seq,
        }
        output.update(fusion_out)
        return output


@dataclass
class PreparedData:
    monthly: Dict[str, np.ndarray]
    windows: Dict[str, np.ndarray]
    context_feature_names: List[str]


class FusionMethodSelector:
    def __init__(
        self,
        aligned_data: Dict[str, np.ndarray],
        output_dir: Path,
        selector_backend: str | None = None,
        frequency: str = "monthly",
        selection_rule: str | None = None,
        stability_lambda: float | None = None,
        seed_list: List[int] | None = None,
        target_mode: str | None = None,
        forecast_horizon_days: int | None = None,
    ) -> None:
        self.aligned_data = aligned_data
        self.output_dir = output_dir
        self.frequency = normalize_frequency(frequency)
        self.output_config = output_config(self.frequency)
        self.index_column = "date" if self.frequency == "daily" else "month"
        session_name = pd.Timestamp.now().strftime("session_%Y%m%d_%H%M%S")
        self.method_runs_dir = output_dir / self.output_config["method_runs_dir"] / session_name
        self.method_runs_dir.mkdir(parents=True, exist_ok=True)
        self.window_builder = WindowBuilder(
            window_lengths=default_window_lengths(self.frequency),
            default_window_length=default_window_length(self.frequency),
        )
        self.dataset_builder = DatasetBuilder()
        self.device = torch.device(FUSION_CONFIG["device"])
        self.seed = FUSION_CONFIG["seed"]
        self.selector_backend = selector_backend or FUSION_CONFIG["selector_backend"]
        self.selection_rule = selection_rule or FUSION_CONFIG.get("selection_rule", "val_rmse")
        if self.selection_rule not in {"val_rmse", "stable_score"}:
            raise ValueError(f"Unsupported selection rule: {self.selection_rule}")
        self.stability_lambda = float(
            FUSION_CONFIG.get("stability_lambda", 0.25) if stability_lambda is None else stability_lambda
        )
        self.seed_list = [int(seed) for seed in (seed_list or FUSION_CONFIG["selector_seed_list"])]
        self.split_strategy = FUSION_CONFIG["selector_split_strategy"]
        self.target_mode = target_mode or FUSION_CONFIG.get("target_mode", "level")
        if self.target_mode not in {"level", "residual", "relative_residual"}:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        self.forecast_horizon_days = int(
            forecast_horizon_days if forecast_horizon_days is not None else FUSION_CONFIG.get("forecast_horizon_days", 7)
        )
        self.timemixer_trainer = TimeMixerSelectorTrainer(self.device, target_mode=self.target_mode)

    def _trailing_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.zeros_like(values, dtype=np.float32)
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            output[idx] = float(np.mean(values[start : idx + 1]))
        return output

    def _trailing_std(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.zeros_like(values, dtype=np.float32)
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            output[idx] = float(np.std(values[start : idx + 1]))
        return output

    def _trailing_median(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.zeros_like(values, dtype=np.float32)
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            output[idx] = float(np.median(values[start : idx + 1]))
        return output

    def _horizon_source_path(self) -> Path:
        return STRUCTURED_DAILY_PATH if self.frequency == "daily" else STRUCTURED_MONTHLY_DERIVED_PATH

    def _resolve_level_labels(
        self,
        labels: np.ndarray,
        index_values: np.ndarray,
    ) -> np.ndarray:
        if self.forecast_horizon_days == 7:
            return labels.astype(np.float32)

        source_path = self._horizon_source_path()
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot build horizon labels because structured source is missing: {source_path}")

        source = pd.read_csv(source_path)
        index_col = "date" if self.frequency == "daily" else "month"
        if index_col not in source.columns:
            raise ValueError(f"Structured source {source_path} does not contain index column {index_col!r}.")
        price_col = REFERENCE_PRICE_COLUMN if REFERENCE_PRICE_COLUMN in source.columns else "reference_brent"
        if price_col not in source.columns:
            raise ValueError(f"Structured source {source_path} does not contain a Brent/reference price column.")

        target_col = f"target_brent_avg_next_{self.forecast_horizon_days}d"
        if target_col not in source.columns:
            source[target_col] = compute_forward_average(source[price_col], horizon=self.forecast_horizon_days)

        target_lookup = {
            str(row[index_col]): float(row[target_col])
            for _, row in source[[index_col, target_col]].dropna(subset=[index_col]).iterrows()
        }
        resolved = np.asarray([target_lookup.get(str(value), np.nan) for value in index_values], dtype=np.float32)
        return resolved

    def _training_labels(self, level_labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if self.target_mode == "residual":
            return (level_labels.astype(np.float32) - reference.astype(np.float32)).astype(np.float32)
        if self.target_mode == "relative_residual":
            reference_values = reference.astype(np.float32)
            return (
                (level_labels.astype(np.float32) - reference_values)
                / np.maximum(np.abs(reference_values), 1e-6)
            ).astype(np.float32)
        return level_labels.astype(np.float32)

    def build_temporal_context(
        self,
        structured: np.ndarray,
        months: np.ndarray,
        missing_flags: np.ndarray,
        feature_names: List[str] | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        feature_names = feature_names or []
        if STEP_RETURN_COLUMN in feature_names:
            return_idx = feature_names.index(STEP_RETURN_COLUMN)
        else:
            return_idx = max(structured.shape[1] - 2, 0)
        if STEP_VOLATILITY_COLUMN in feature_names:
            volatility_idx = feature_names.index(STEP_VOLATILITY_COLUMN)
        else:
            volatility_idx = max(structured.shape[1] - 1, 0)
        if REFERENCE_PRICE_COLUMN in feature_names:
            brent_idx = feature_names.index(REFERENCE_PRICE_COLUMN)
        else:
            brent_idx = 0

        step_return = structured[:, return_idx].astype(np.float32)
        volatility = structured[:, volatility_idx].astype(np.float32)
        brent_level = structured[:, brent_idx].astype(np.float32)
        abs_return = np.abs(step_return)

        timestamps = pd.to_datetime(np.asarray(months).astype(str), errors="coerce")
        month_numbers = timestamps.month.to_numpy(dtype=np.float32)
        month_angles = 2 * np.pi * (month_numbers - 1.0) / 12.0
        month_sin = np.sin(month_angles).astype(np.float32)
        month_cos = np.cos(month_angles).astype(np.float32)
        weekday_numbers = timestamps.dayofweek.fillna(0).to_numpy(dtype=np.float32)
        weekday_angles = 2 * np.pi * weekday_numbers / 7.0
        weekday_sin = np.sin(weekday_angles).astype(np.float32)
        weekday_cos = np.cos(weekday_angles).astype(np.float32)
        dayofyear_numbers = timestamps.dayofyear.fillna(1).to_numpy(dtype=np.float32)
        dayofyear_angles = 2 * np.pi * (dayofyear_numbers - 1.0) / 366.0
        dayofyear_sin = np.sin(dayofyear_angles).astype(np.float32)
        dayofyear_cos = np.cos(dayofyear_angles).astype(np.float32)

        return_ma3 = self._trailing_mean(step_return, 3)
        volatility_ma3 = self._trailing_mean(volatility, 3)
        volatility_std3 = self._trailing_std(volatility, 3)
        abs_return_std3 = self._trailing_std(abs_return, 3)
        high_vol_regime = (volatility >= self._trailing_median(volatility, 6)).astype(np.float32)
        shock_flag = (
            (abs_return >= (self._trailing_mean(abs_return, 3) + abs_return_std3))
            | (volatility >= (volatility_ma3 + volatility_std3))
        ).astype(np.float32)

        context = np.stack(
            [
                month_sin,
                month_cos,
                weekday_sin,
                weekday_cos,
                dayofyear_sin,
                dayofyear_cos,
                brent_level,
                step_return,
                volatility,
                return_ma3,
                volatility_ma3,
                high_vol_regime,
                shock_flag,
                missing_flags[:, 0].astype(np.float32),
                missing_flags[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        names = [
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
            "dayofyear_sin",
            "dayofyear_cos",
            "brent_level",
            "step_return",
            "volatility",
            "return_ma3",
            "volatility_ma3",
            "high_vol_regime",
            "shock_flag",
            "text_missing_flag",
            "image_missing_flag",
        ]
        return context, names

    def prepare_inputs(self, window_length: int) -> PreparedData:
        structured = self.aligned_data["structured_features"]
        text = self.aligned_data["text_embeddings"]
        image = self.aligned_data["image_embeddings"]
        reference = self.aligned_data.get("reference")
        months = self.aligned_data["months"]
        labels = self._resolve_level_labels(self.aligned_data["labels"], months)
        train_labels = self._training_labels(labels, reference)
        missing_flags = self.aligned_data["missing_flags"]
        text_counts = self.aligned_data["text_counts"]
        image_counts = self.aligned_data["image_counts"]
        feature_names = self.aligned_data.get("structured_feature_names", [])
        context, context_feature_names = self.build_temporal_context(structured, months, missing_flags, feature_names)

        structured_windows, y, window_months = self.window_builder.build_windows_with_months(
            structured, train_labels, months, window_length
        )
        text_windows, _ = self.window_builder.build_windows(text, train_labels, window_length)
        image_windows, _ = self.window_builder.build_windows(image, train_labels, window_length)
        context_windows, _ = self.window_builder.build_windows(context, train_labels, window_length)
        missing_flag_windows, _ = self.window_builder.build_windows(missing_flags, train_labels, window_length)
        text_count_windows, _ = self.window_builder.build_windows(text_counts.reshape(-1, 1), train_labels, window_length)
        image_count_windows, _ = self.window_builder.build_windows(image_counts.reshape(-1, 1), train_labels, window_length)
        reference_windows, _ = self.window_builder.build_windows(reference.reshape(-1, 1), train_labels, window_length)

        monthly = {
            "structured": structured,
            "text": text,
            "image": image,
            "context": context,
            "labels": labels,
            "train_labels": train_labels,
            "reference": reference,
            "months": months,
            "missing_flags": missing_flags,
            "text_counts": text_counts,
            "image_counts": image_counts,
        }
        windows = {
            "structured": structured_windows,
            "text": text_windows,
            "image": image_windows,
            "context": context_windows,
            "labels": y,
            "reference": reference_windows[:, -1, 0],
            "months": window_months,
            "missing_flags": missing_flag_windows,
            "text_counts": text_count_windows.squeeze(-1),
            "image_counts": image_count_windows.squeeze(-1),
        }
        return PreparedData(monthly=monthly, windows=windows, context_feature_names=context_feature_names)

    def split_windows(self, windows: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        total_samples = len(windows["labels"])
        train_end = int(total_samples * self.dataset_builder.train_ratio)
        valid_end = int(total_samples * (self.dataset_builder.train_ratio + self.dataset_builder.valid_ratio))

        split_indices = {
            "train": slice(0, train_end),
            "valid": slice(train_end, valid_end),
            "test": slice(valid_end, total_samples),
        }
        split_data: Dict[str, Dict[str, np.ndarray]] = {}
        for split, split_slice in split_indices.items():
            split_data[split] = {
                name: values[split_slice] for name, values in windows.items()
            }
        return split_data

    def _slice_split(self, windows: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
        return {name: values[start:end] for name, values in windows.items()}

    def build_rolling_origin_plan(self, windows: Dict[str, np.ndarray]) -> Dict[str, object]:
        total_samples = len(windows["labels"])
        val_size, test_size = split_sizes(self.frequency)
        rolling_folds = FUSION_CONFIG["selector_rolling_folds"]

        pre_test_end = total_samples - test_size
        first_fold_train_end = pre_test_end - rolling_folds * val_size
        if first_fold_train_end <= 0:
            raise ValueError(
                f"Not enough samples ({total_samples}) for rolling split with "
                f"{rolling_folds} folds, val_size={val_size}, test_size={test_size}."
            )

        validation_folds = []
        for fold_idx in range(rolling_folds):
            train_end = first_fold_train_end + fold_idx * val_size
            valid_start = train_end
            valid_end = valid_start + val_size
            validation_folds.append(
                {
                    "fold_index": fold_idx + 1,
                    "train": self._slice_split(windows, 0, train_end),
                    "valid": self._slice_split(windows, valid_start, valid_end),
                }
            )

        final_valid_start = pre_test_end - val_size
        final_valid_end = pre_test_end
        final_plan = {
            "train": self._slice_split(windows, 0, final_valid_start),
            "valid": self._slice_split(windows, final_valid_start, final_valid_end),
            "test": self._slice_split(windows, pre_test_end, total_samples),
        }
        return {
            "validation_folds": validation_folds,
            "final_plan": final_plan,
            "val_size": val_size,
            "test_size": test_size,
            "rolling_folds": rolling_folds,
        }

    def build_split_plan(self, windows: Dict[str, np.ndarray]) -> Dict[str, object]:
        if self.split_strategy == "rolling_origin":
            return self.build_rolling_origin_plan(windows)
        split_data = self.split_windows(windows)
        return {
            "validation_folds": [],
            "final_plan": split_data,
            "val_size": len(split_data["valid"]["labels"]),
            "test_size": len(split_data["test"]["labels"]),
            "rolling_folds": 0,
        }

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _metrics(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> Dict[str, float]:
        metrics = {
            "mse": mse(pred, true),
            "rmse": rmse(pred, true),
            "mae": mae(pred, true),
        }
        if reference is not None:
            if self.target_mode == "residual":
                metrics["direction_acc"] = float(np.mean(np.sign(pred) == np.sign(true)))
            else:
                metrics["direction_acc"] = direction_accuracy(pred, true, reference)
        return metrics

    def _evaluate_split(self, model: StructuredFusionRegressor, split_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            preds = model(
                self._to_tensor(split_dict["structured"]),
                self._to_tensor(split_dict["text"]),
                self._to_tensor(split_dict["image"]),
                self._to_tensor(split_dict["context"]),
            ).detach().cpu().numpy()
        reference = split_dict["reference"]
        return self._metrics(preds, split_dict["labels"], reference)

    def _fit_structured_scaler(self, monthly_structured: np.ndarray) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(monthly_structured)
        return scaler

    def _transform_structured_windows(self, split_data: Dict[str, Dict[str, np.ndarray]], scaler: StandardScaler):
        for split in split_data.values():
            raw_shape = split["structured"].shape
            flat = split["structured"].reshape(-1, raw_shape[-1])
            split["structured"] = scaler.transform(flat).reshape(raw_shape)

    def _transform_structured_monthly(self, monthly_structured: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        return scaler.transform(monthly_structured)

    def _transform_windows_structured(self, windows: Dict[str, np.ndarray], scaler: StandardScaler) -> Dict[str, np.ndarray]:
        transformed = deepcopy(windows)
        raw_shape = transformed["structured"].shape
        flat = transformed["structured"].reshape(-1, raw_shape[-1])
        transformed["structured"] = scaler.transform(flat).reshape(raw_shape)
        return transformed

    def _aggregate_validation_metrics(self, validation_results: List[Dict[str, object]]) -> Dict[str, float]:
        metric_names = sorted({key for fold in validation_results for key in fold["val"].keys()})
        return {
            metric_name: float(np.mean([fold["val"][metric_name] for fold in validation_results]))
            for metric_name in metric_names
        }

    def _extract_monthly_aux(self, monthly_out: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        aux: Dict[str, np.ndarray] = {}
        if "gate" in monthly_out:
            gate_mean = monthly_out["gate"].detach().cpu().numpy().mean(axis=-1).squeeze(1)
            aux["gate_mean"] = gate_mean
            aux["gate_text_share"] = gate_mean
            aux["gate_image_share"] = 1.0 - gate_mean
        if "modality_weights" in monthly_out:
            weights = monthly_out["modality_weights"].detach().cpu().numpy().squeeze(1)
            aux["text_weight"] = weights[:, 0]
            aux["image_weight"] = weights[:, 1]
        return aux

    def _save_method_artifacts(
        self,
        method_dir: Path,
        method_id: str,
        monthly_features: np.ndarray,
        structured_proj: np.ndarray,
        fused_text_image: np.ndarray,
        labels: np.ndarray,
        months: np.ndarray,
        missing_flags: np.ndarray,
        text_counts: np.ndarray,
        image_counts: np.ndarray,
        monthly_context: np.ndarray,
        context_feature_names: List[str],
        scaler: StandardScaler,
        metrics: Dict[str, object],
        monthly_aux: Dict[str, np.ndarray] | None = None,
    ) -> None:
        np.save(method_dir / self.output_config["features_file"], monthly_features)
        np.save(method_dir / self.output_config["labels_file"], labels)
        np.save(method_dir / self.output_config["months_file"], months)
        np.save(method_dir / self.output_config["missing_flags_file"], missing_flags)
        np.save(method_dir / "fusion_months.npy", months)
        np.savez(method_dir / self.output_config["scaler_file"], mean=scaler.mean_, scale=scaler.scale_)

        method_df = pd.DataFrame(
            {
                self.index_column: months,
                "label": labels,
                "text_count": text_counts,
                "image_count": image_counts,
                "text_missing_flag": missing_flags[:, 0],
                "image_missing_flag": missing_flags[:, 1],
                "structured_projected": [row.tolist() for row in structured_proj],
                "fused_text_image": [row.tolist() for row in fused_text_image],
                "fusion_feature": [row.tolist() for row in monthly_features],
                "temporal_context": [row.tolist() for row in monthly_context],
                "method_id": method_id,
            }
        )
        if monthly_aux:
            for key, values in monthly_aux.items():
                method_df[key] = values
        method_df.to_csv(method_dir / self.output_config["processed_file"], index=False, encoding="utf-8")

        metrics["context_feature_names"] = context_feature_names
        with open(method_dir / self.output_config["metrics_file"], "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)

    def _copy_run_artifacts(self, source_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for output_name in [
            self.output_config["features_file"],
            self.output_config["processed_file"],
            self.output_config["labels_file"],
            self.output_config["months_file"],
            self.output_config["missing_flags_file"],
            self.output_config["scaler_file"],
            self.output_config["metrics_file"],
        ]:
            shutil.copy2(source_dir / output_name, target_dir / output_name)

    def _aggregate_seed_results(self, seed_results: List[Dict[str, object]]) -> Dict[str, float]:
        metric_names = [
            "val_rmse",
            "val_mae",
            "val_mse",
            "val_direction_acc",
            "test_rmse",
            "test_mae",
            "test_mse",
            "test_direction_acc",
            "train_rmse",
            "train_mae",
            "train_mse",
            "train_direction_acc",
        ]
        aggregate: Dict[str, float] = {}
        for metric_name in metric_names:
            metric_values = [float(result[metric_name]) for result in seed_results if metric_name in result]
            if not metric_values:
                continue
            aggregate[metric_name] = float(np.mean(metric_values))
            aggregate[f"{metric_name}_std"] = float(np.std(metric_values))
        aggregate["seed_count"] = len(seed_results)
        return aggregate

    def _select_representative_seed(self, seed_results: List[Dict[str, object]]) -> Dict[str, object]:
        target_val_rmse = float(np.mean([result["val_rmse"] for result in seed_results]))
        return sorted(
            seed_results,
            key=lambda item: (
                abs(float(item["val_rmse"]) - target_val_rmse),
                float(item["test_rmse"]),
                int(item["seed"]),
            ),
        )[0]

    def _method_traits(self, stage: str, method: str) -> Dict[str, str]:
        simplicity = {
            "early": "high",
            "intermediate": "medium",
            "late": "medium_low",
        }[stage]
        explainability = "medium"
        if method in {"weighted_sum_fixed", "weighted_sum_learnable", "gated_fusion", "temporal_attention_fusion", "gru_gate"}:
            explainability = "high"
        return {
            "simplicity": simplicity,
            "explainability": explainability,
        }

    def _select_generalization_best(self, results: List[Dict[str, object]]) -> Dict[str, object]:
        return sorted(
            results,
            key=lambda item: (
                float(item["test_rmse"]),
                float(item["test_mae"]),
                -float(item.get("test_direction_acc", 0.0)),
                float(item["val_rmse"]),
                STAGE_PRIORITY[str(item["stage"])],
                str(item["method"]),
            ),
        )[0]

    def _stable_score(self, item: Dict[str, object]) -> float:
        return float(item["val_rmse"]) + self.stability_lambda * float(item.get("val_rmse_std", 0.0))

    def _selection_score(self, item: Dict[str, object]) -> float:
        if self.selection_rule == "stable_score":
            return self._stable_score(item)
        return float(item["val_rmse"])

    def _select_research_mainline(self, results: List[Dict[str, object]]) -> Dict[str, object]:
        preferred_stage = FUSION_CONFIG["research_mainline_preferred_stage"]
        stage_candidates = [item for item in results if item["stage"] == preferred_stage]
        candidate_pool = stage_candidates or results
        return sorted(
            candidate_pool,
            key=lambda item: (
                self._selection_score(item),
                float(item["val_rmse"]),
                float(item.get("val_rmse_std", 0.0)),
                float(item["test_rmse"]),
                -float(item.get("test_direction_acc", 0.0)),
                str(item["method"]),
            ),
        )[0]

    def _train_method_multi_seed(
        self,
        stage: str,
        method: str,
        prepared: PreparedData,
        alpha: float,
        window_length: int,
        fused_dim: int,
    ) -> Dict[str, object]:
        seed_list = list(self.seed_list)
        method_dir = self.method_runs_dir / f"{stage}__{method}"
        seed_runs_dir = method_dir / "seed_runs"
        seed_runs_dir.mkdir(parents=True, exist_ok=True)

        original_seed = self.seed
        seed_results: List[Dict[str, object]] = []
        for seed in seed_list:
            self.seed = int(seed)
            run_result = self._train_single_method(stage, method, prepared, alpha, window_length, fused_dim)
            run_result["seed"] = int(seed)
            run_result["seed_dir"] = str(seed_runs_dir / f"seed_{seed}")
            self._copy_run_artifacts(Path(run_result["method_dir"]), seed_runs_dir / f"seed_{seed}")
            seed_results.append(run_result)

        self.seed = original_seed

        aggregate_metrics = self._aggregate_seed_results(seed_results)
        stable_score = float(aggregate_metrics["val_rmse"]) + self.stability_lambda * float(
            aggregate_metrics.get("val_rmse_std", 0.0)
        )
        representative = self._select_representative_seed(seed_results)
        representative_dir = Path(str(representative["seed_dir"]))
        self._copy_run_artifacts(representative_dir, method_dir)

        aggregate_payload = {
            "method_id": f"{stage}.{method}",
            "selector_backend": self.selector_backend,
            "selector_split_strategy": self.split_strategy,
            "window_length": window_length,
            "fused_dim": fused_dim,
            "selection_rule": self.selection_rule,
            "stability_lambda": self.stability_lambda,
            "target_mode": self.target_mode,
            "forecast_horizon_days": self.forecast_horizon_days,
            "stable_score": stable_score,
            "context_feature_names": prepared.context_feature_names,
            "seed_list": [int(seed) for seed in seed_list],
            "representative_seed": int(representative["seed"]),
            "aggregate": aggregate_metrics,
            "per_seed": [
                {
                    "seed": int(result["seed"]),
                    "method_dir": result["seed_dir"],
                    "train": {
                        "mse": float(result["train_mse"]),
                        "rmse": float(result["train_rmse"]),
                        "mae": float(result["train_mae"]),
                        "direction_acc": float(result.get("train_direction_acc", 0.0)),
                    },
                    "val": {
                        "mse": float(result["val_mse"]),
                        "rmse": float(result["val_rmse"]),
                        "mae": float(result["val_mae"]),
                        "direction_acc": float(result.get("val_direction_acc", 0.0)),
                    },
                    "test": {
                        "mse": float(result["test_mse"]),
                        "rmse": float(result["test_rmse"]),
                        "mae": float(result["test_mae"]),
                        "direction_acc": float(result.get("test_direction_acc", 0.0)),
                    },
                    "best_epoch": int(result["best_epoch"]),
                }
                for result in seed_results
            ],
        }
        with open(method_dir / self.output_config["metrics_file"], "w", encoding="utf-8") as fh:
            json.dump(aggregate_payload, fh, indent=2, ensure_ascii=False)

        traits = self._method_traits(stage, method)
        return {
            "stage": stage,
            "method": method,
            "method_id": f"{stage}.{method}",
            "selector_backend": self.selector_backend,
            "selection_rule": self.selection_rule,
            "stability_lambda": self.stability_lambda,
            "target_mode": self.target_mode,
            "forecast_horizon_days": self.forecast_horizon_days,
            "stable_score": stable_score,
            "val_rmse": aggregate_metrics["val_rmse"],
            "val_rmse_std": aggregate_metrics.get("val_rmse_std", 0.0),
            "val_mae": aggregate_metrics["val_mae"],
            "val_mae_std": aggregate_metrics.get("val_mae_std", 0.0),
            "val_mse": aggregate_metrics["val_mse"],
            "val_direction_acc": aggregate_metrics.get("val_direction_acc", 0.0),
            "val_direction_acc_std": aggregate_metrics.get("val_direction_acc_std", 0.0),
            "test_rmse": aggregate_metrics["test_rmse"],
            "test_rmse_std": aggregate_metrics.get("test_rmse_std", 0.0),
            "test_mae": aggregate_metrics["test_mae"],
            "test_mae_std": aggregate_metrics.get("test_mae_std", 0.0),
            "test_mse": aggregate_metrics["test_mse"],
            "test_direction_acc": aggregate_metrics.get("test_direction_acc", 0.0),
            "test_direction_acc_std": aggregate_metrics.get("test_direction_acc_std", 0.0),
            "train_rmse": aggregate_metrics.get("train_rmse", 0.0),
            "train_rmse_std": aggregate_metrics.get("train_rmse_std", 0.0),
            "train_mae": aggregate_metrics.get("train_mae", 0.0),
            "train_mae_std": aggregate_metrics.get("train_mae_std", 0.0),
            "train_mse": aggregate_metrics.get("train_mse", 0.0),
            "train_direction_acc": aggregate_metrics.get("train_direction_acc", 0.0),
            "train_direction_acc_std": aggregate_metrics.get("train_direction_acc_std", 0.0),
            "seed_count": aggregate_metrics["seed_count"],
            "representative_seed": int(representative["seed"]),
            "best_epoch": int(representative["best_epoch"]),
            "method_dir": to_project_relative(method_dir),
            **traits,
        }

    def _train_single_method_sequence_pool(
        self,
        stage: str,
        method: str,
        prepared: PreparedData,
        alpha: float,
        window_length: int,
        fused_dim: int,
    ) -> Dict[str, object]:
        set_seed(self.seed)
        scaler = self._fit_structured_scaler(prepared.monthly["structured"])
        windows_scaled = self._transform_windows_structured(prepared.windows, scaler)
        split_plan = self.build_split_plan(windows_scaled)

        validation_results = []
        if self.split_strategy == "rolling_origin":
            for fold in split_plan["validation_folds"]:
                set_seed(self.seed + int(fold["fold_index"]))
                model = StructuredFusionRegressor(
                    stage=stage,
                    method=method,
                    structured_dim=prepared.monthly["structured"].shape[-1],
                    text_dim=prepared.monthly["text"].shape[-1],
                    image_dim=prepared.monthly["image"].shape[-1],
                    context_dim=prepared.monthly["context"].shape[-1],
                    fused_dim=fused_dim,
                    alpha=alpha,
                ).to(self.device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=FUSION_CONFIG["selector_learning_rate"],
                    weight_decay=FUSION_CONFIG["selector_weight_decay"],
                )
                loss_fn = nn.MSELoss()

                best_state = deepcopy(model.state_dict())
                best_val_rmse = math.inf
                best_epoch = 0
                best_val_metrics: Dict[str, float] | None = None
                patience_counter = 0

                for epoch in range(FUSION_CONFIG["selector_max_epochs"]):
                    model.train()
                    optimizer.zero_grad()
                    output = model(
                        self._to_tensor(fold["train"]["structured"]),
                        self._to_tensor(fold["train"]["text"]),
                        self._to_tensor(fold["train"]["image"]),
                        self._to_tensor(fold["train"]["context"]),
                        return_aux=True,
                    )
                    pred = output["prediction"]
                    target = self._to_tensor(fold["train"]["labels"])
                    loss = loss_fn(pred, target) + model.auxiliary_loss(output)
                    loss.backward()
                    optimizer.step()

                    fold_val_metrics = self._evaluate_split(model, fold["valid"])
                    if fold_val_metrics["rmse"] < best_val_rmse - 1e-12:
                        best_val_rmse = fold_val_metrics["rmse"]
                        best_epoch = epoch + 1
                        best_val_metrics = fold_val_metrics
                        best_state = deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= FUSION_CONFIG["selector_patience"]:
                            break

                model.load_state_dict(best_state)
                validation_results.append(
                    {
                        "fold_index": fold["fold_index"],
                        "train": self._evaluate_split(model, fold["train"]),
                        "val": best_val_metrics or self._evaluate_split(model, fold["valid"]),
                        "best_epoch": best_epoch,
                    }
                )
            val_metrics = self._aggregate_validation_metrics(validation_results)
        else:
            validation_results = []
            val_metrics = None

        final_model = StructuredFusionRegressor(
            stage=stage,
            method=method,
            structured_dim=prepared.monthly["structured"].shape[-1],
            text_dim=prepared.monthly["text"].shape[-1],
            image_dim=prepared.monthly["image"].shape[-1],
            context_dim=prepared.monthly["context"].shape[-1],
            fused_dim=fused_dim,
            alpha=alpha,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=FUSION_CONFIG["selector_learning_rate"],
            weight_decay=FUSION_CONFIG["selector_weight_decay"],
        )
        loss_fn = nn.MSELoss()
        best_state = deepcopy(final_model.state_dict())
        best_val_rmse = math.inf
        best_epoch = 0
        best_final_val_metrics: Dict[str, float] | None = None
        patience_counter = 0

        final_train = split_plan["final_plan"]["train"]
        final_valid = split_plan["final_plan"]["valid"]
        final_test = split_plan["final_plan"]["test"]

        for epoch in range(FUSION_CONFIG["selector_max_epochs"]):
            final_model.train()
            optimizer.zero_grad()
            output = final_model(
                self._to_tensor(final_train["structured"]),
                self._to_tensor(final_train["text"]),
                self._to_tensor(final_train["image"]),
                self._to_tensor(final_train["context"]),
                return_aux=True,
            )
            pred = output["prediction"]
            target = self._to_tensor(final_train["labels"])
            loss = loss_fn(pred, target) + final_model.auxiliary_loss(output)
            loss.backward()
            optimizer.step()

            current_val_metrics = self._evaluate_split(final_model, final_valid)
            if current_val_metrics["rmse"] < best_val_rmse - 1e-12:
                best_val_rmse = current_val_metrics["rmse"]
                best_epoch = epoch + 1
                best_final_val_metrics = current_val_metrics
                best_state = deepcopy(final_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= FUSION_CONFIG["selector_patience"]:
                    break

        final_model.load_state_dict(best_state)

        train_metrics = self._evaluate_split(final_model, final_train)
        test_metrics = self._evaluate_split(final_model, final_test)
        if val_metrics is None:
            val_metrics = best_final_val_metrics or self._evaluate_split(final_model, final_valid)

        monthly_structured_scaled = self._transform_structured_monthly(prepared.monthly["structured"], scaler)
        final_model.eval()
        with torch.no_grad():
            monthly_out = final_model(
                self._to_tensor(monthly_structured_scaled[:, None, :]),
                self._to_tensor(prepared.monthly["text"][:, None, :]),
                self._to_tensor(prepared.monthly["image"][:, None, :]),
                self._to_tensor(prepared.monthly["context"][:, None, :]),
                return_aux=True,
            )

        final_monthly = monthly_out["final_seq"].detach().cpu().numpy().squeeze(1)
        structured_proj = monthly_out["structured_proj"].detach().cpu().numpy().squeeze(1)
        fused_text_image = monthly_out["fused_text_image"].detach().cpu().numpy().squeeze(1)
        monthly_aux = self._extract_monthly_aux(monthly_out)

        method_id = f"{stage}.{method}"
        method_dir = self.method_runs_dir / f"{stage}__{method}"
        method_dir.mkdir(parents=True, exist_ok=True)

        self._save_method_artifacts(
            method_dir=method_dir,
            method_id=method_id,
            monthly_features=final_monthly,
            structured_proj=structured_proj,
            fused_text_image=fused_text_image,
            labels=prepared.monthly["labels"],
            months=prepared.monthly["months"],
            missing_flags=prepared.monthly["missing_flags"],
            text_counts=prepared.monthly["text_counts"],
            image_counts=prepared.monthly["image_counts"],
            monthly_context=prepared.monthly["context"],
            context_feature_names=prepared.context_feature_names,
            scaler=scaler,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "best_epoch": best_epoch,
                "window_length": window_length,
                "fused_dim": fused_dim,
                "method_id": method_id,
                "selector_backend": self.selector_backend,
                "selector_split_strategy": self.split_strategy,
                "target_mode": self.target_mode,
                "forecast_horizon_days": self.forecast_horizon_days,
                "rolling_validation": validation_results,
            },
            monthly_aux=monthly_aux,
        )

        return {
            "stage": stage,
            "method": method,
            "method_id": method_id,
            "selector_backend": self.selector_backend,
            "target_mode": self.target_mode,
            "forecast_horizon_days": self.forecast_horizon_days,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_mse": train_metrics["mse"],
            "train_direction_acc": train_metrics.get("direction_acc", 0.0),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_direction_acc": val_metrics.get("direction_acc", 0.0),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_direction_acc": test_metrics.get("direction_acc", 0.0),
            "best_epoch": best_epoch,
            "method_dir": str(method_dir),
        }

    def _train_single_method_timemixer(
        self,
        stage: str,
        method: str,
        prepared: PreparedData,
        alpha: float,
        window_length: int,
        fused_dim: int,
    ) -> Dict[str, object]:
        set_seed(self.seed)

        scaler = self._fit_structured_scaler(prepared.monthly["structured"])
        windows_scaled = self._transform_windows_structured(prepared.windows, scaler)
        split_plan = self.build_split_plan(windows_scaled)

        validation_results = []
        if self.split_strategy == "rolling_origin":
            for fold in split_plan["validation_folds"]:
                set_seed(self.seed + int(fold["fold_index"]))
                model = FusionTimeMixerForecaster(
                    stage=stage,
                    method=method,
                    structured_dim=prepared.monthly["structured"].shape[-1],
                    text_dim=prepared.monthly["text"].shape[-1],
                    image_dim=prepared.monthly["image"].shape[-1],
                    context_dim=prepared.monthly["context"].shape[-1],
                    fused_dim=fused_dim,
                    alpha=alpha,
                    seq_len=window_length,
                    frequency=self.frequency,
                ).to(self.device)
                fold_summary = self.timemixer_trainer.fit(model, fold["train"], fold["valid"])
                validation_results.append(
                    {
                        "fold_index": fold["fold_index"],
                        "train": fold_summary["train"],
                        "val": fold_summary["val"],
                        "best_epoch": fold_summary["best_epoch"],
                    }
                )
            val_metrics = self._aggregate_validation_metrics(validation_results)
        else:
            validation_results = []
            val_metrics = None

        final_model = FusionTimeMixerForecaster(
            stage=stage,
            method=method,
            structured_dim=prepared.monthly["structured"].shape[-1],
            text_dim=prepared.monthly["text"].shape[-1],
            image_dim=prepared.monthly["image"].shape[-1],
            context_dim=prepared.monthly["context"].shape[-1],
            fused_dim=fused_dim,
            alpha=alpha,
            seq_len=window_length,
            frequency=self.frequency,
        ).to(self.device)

        final_summary = self.timemixer_trainer.fit(
            final_model,
            split_plan["final_plan"]["train"],
            split_plan["final_plan"]["valid"],
            split_plan["final_plan"]["test"],
        )
        train_metrics = final_summary["train"]
        test_metrics = final_summary["test"]
        best_epoch = final_summary["best_epoch"]
        if val_metrics is None:
            val_metrics = final_summary["val"]

        monthly_structured_scaled = self._transform_structured_monthly(prepared.monthly["structured"], scaler)
        final_model.eval()
        with torch.no_grad():
            monthly_out = final_model.encode_sequence(
                self._to_tensor(monthly_structured_scaled[:, None, :]),
                self._to_tensor(prepared.monthly["text"][:, None, :]),
                self._to_tensor(prepared.monthly["image"][:, None, :]),
                self._to_tensor(prepared.monthly["context"][:, None, :]),
                return_aux=True,
            )

        final_monthly = monthly_out["final_seq"].detach().cpu().numpy().squeeze(1)
        structured_proj = monthly_out["structured_proj"].detach().cpu().numpy().squeeze(1)
        fused_text_image = monthly_out["fused_text_image"].detach().cpu().numpy().squeeze(1)
        monthly_aux = self._extract_monthly_aux(monthly_out)

        method_id = f"{stage}.{method}"
        method_dir = self.method_runs_dir / f"{stage}__{method}"
        method_dir.mkdir(parents=True, exist_ok=True)

        self._save_method_artifacts(
            method_dir=method_dir,
            method_id=method_id,
            monthly_features=final_monthly,
            structured_proj=structured_proj,
            fused_text_image=fused_text_image,
            labels=prepared.monthly["labels"],
            months=prepared.monthly["months"],
            missing_flags=prepared.monthly["missing_flags"],
            text_counts=prepared.monthly["text_counts"],
            image_counts=prepared.monthly["image_counts"],
            monthly_context=prepared.monthly["context"],
            context_feature_names=prepared.context_feature_names,
            scaler=scaler,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "best_epoch": best_epoch,
                "window_length": window_length,
                "fused_dim": fused_dim,
                "method_id": method_id,
                "selector_backend": self.selector_backend,
                "selector_split_strategy": self.split_strategy,
                "target_mode": self.target_mode,
                "forecast_horizon_days": self.forecast_horizon_days,
                "timemixer": final_summary["timemixer_config"],
                "rolling_validation": validation_results,
            },
            monthly_aux=monthly_aux,
        )

        return {
            "stage": stage,
            "method": method,
            "method_id": method_id,
            "selector_backend": self.selector_backend,
            "target_mode": self.target_mode,
            "forecast_horizon_days": self.forecast_horizon_days,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_mse": train_metrics["mse"],
            "train_direction_acc": train_metrics.get("direction_acc", 0.0),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_direction_acc": val_metrics.get("direction_acc", 0.0),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_direction_acc": test_metrics.get("direction_acc", 0.0),
            "best_epoch": best_epoch,
            "method_dir": str(method_dir),
        }

    def _train_single_method(
        self,
        stage: str,
        method: str,
        prepared: PreparedData,
        alpha: float,
        window_length: int,
        fused_dim: int,
    ) -> Dict[str, object]:
        if self.selector_backend == "timemixer":
            return self._train_single_method_timemixer(stage, method, prepared, alpha, window_length, fused_dim)
        return self._train_single_method_sequence_pool(stage, method, prepared, alpha, window_length, fused_dim)

    def _sort_results(self, results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        def sort_key(item: Dict[str, object]):
            return (
                self._selection_score(item),
                float(item["val_rmse"]),
                float(item.get("val_rmse_std", 0.0)),
                float(item["val_mae"]),
                STAGE_PRIORITY[str(item["stage"])],
                str(item["method"]),
            )

        return sorted(results, key=sort_key)

    def _write_selection_outputs(
        self,
        sorted_results: List[Dict[str, object]],
        best_result: Dict[str, object],
        prepared: PreparedData,
        window_length: int,
        fused_dim: int,
    ) -> None:
        generalization_best = self._select_generalization_best(sorted_results)
        research_mainline = self._select_research_mainline(sorted_results)

        for row in sorted_results:
            row["stable_score"] = self._stable_score(row)
            row["selection_rule"] = self.selection_rule
            row["stability_lambda"] = self.stability_lambda
            row["is_current_best"] = row["method_id"] == best_result["method_id"]
            row["is_generalization_best"] = row["method_id"] == generalization_best["method_id"]
            row["is_research_mainline"] = row["method_id"] == research_mainline["method_id"]

        selection_df = pd.DataFrame(sorted_results)
        selection_df.to_csv(
            self.output_dir / self.output_config["selection_csv"],
            index=False,
            encoding="utf-8",
        )

        best_choice = {
            "selected_method": best_result["method_id"],
            "selection_metric": self.selection_rule,
            "selection_rule": self.selection_rule,
            "stability_lambda": self.stability_lambda,
            "stable_score": float(best_result.get("stable_score", self._stable_score(best_result))),
            "selector_backend": self.selector_backend,
            "selector_split_strategy": self.split_strategy,
            "target_mode": self.target_mode,
            "forecast_horizon_days": self.forecast_horizon_days,
            "seed_list": list(self.seed_list),
            "seed_count": int(best_result.get("seed_count", 1)),
            "val_rmse": float(best_result["val_rmse"]),
            "val_rmse_std": float(best_result.get("val_rmse_std", 0.0)),
            "val_mae": float(best_result["val_mae"]),
            "val_mae_std": float(best_result.get("val_mae_std", 0.0)),
            "val_mse": float(best_result["val_mse"]),
            "val_direction_acc": float(best_result.get("val_direction_acc", 0.0)),
            "val_direction_acc_std": float(best_result.get("val_direction_acc_std", 0.0)),
            "test_rmse": float(best_result["test_rmse"]),
            "test_rmse_std": float(best_result.get("test_rmse_std", 0.0)),
            "test_mae": float(best_result["test_mae"]),
            "test_mae_std": float(best_result.get("test_mae_std", 0.0)),
            "test_mse": float(best_result["test_mse"]),
            "test_direction_acc": float(best_result.get("test_direction_acc", 0.0)),
            "test_direction_acc_std": float(best_result.get("test_direction_acc_std", 0.0)),
            "window_length": window_length,
            "fused_dim": fused_dim,
            "context_feature_names": prepared.context_feature_names,
            "generalization_best": generalization_best["method_id"],
            "research_mainline": research_mainline["method_id"],
            "reason": (
                "Rank candidate fusion methods without using test metrics. "
                "The default stable_score is val_rmse + stability_lambda * val_rmse_std; "
                "test metrics are diagnostic only."
            ),
        }
        with open(self.output_dir / self.output_config["best_choice_json"], "w", encoding="utf-8") as fh:
            json.dump(best_choice, fh, indent=2, ensure_ascii=False)

        lines = [
            "# Fusion Work Record",
            "",
            "## Data Summary",
            f"- Sample count: {len(prepared.monthly['months'])}",
            f"- Text dimension: {prepared.monthly['text'].shape[-1]}",
            f"- Image dimension: {prepared.monthly['image'].shape[-1]}",
            f"- Structured dimension: {prepared.monthly['structured'].shape[-1]}",
            f"- Context dimension: {prepared.monthly['context'].shape[-1]}",
            f"- Total text documents: {int(prepared.monthly['text_counts'].sum())}",
            f"- Total images: {int(prepared.monthly['image_counts'].sum())}",
            f"- Seed count: {len(self.seed_list)}",
            "",
            "## Context Features",
        ]
        lines.extend([f"- `{name}`" for name in prepared.context_feature_names])
        lines.extend(
            [
                "",
                "## Selection Setup",
                f"- Window length: {window_length}",
                f"- Fused dimension: {fused_dim}",
                f"- Selector backend: `{self.selector_backend}`",
                f"- Split strategy: `{self.split_strategy}`",
                f"- Selection rule: `{self.selection_rule}`",
                f"- Stability lambda: `{self.stability_lambda}`",
                f"- Random seeds: `{self.seed_list}`",
                "- Rule A: rank by the configured validation-only selection score.",
                "- Rule B: report test metrics, direction accuracy, and seed stability as diagnostics only.",
                "- Rule C: mark both the generalization best and the research mainline.",
                "",
                "## Method Comparison",
                "",
                "| Method | stable_score | val_rmse(mean+-std) | val_mae(mean+-std) | val_dir_acc | test_rmse(mean+-std) | test_mae(mean+-std) | test_dir_acc | simplicity | explainability |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in sorted_results:
            lines.append(
                f"| {row['method_id']} | {row.get('stable_score', self._stable_score(row)):.6f} | "
                f"{row['val_rmse']:.6f} +- {row.get('val_rmse_std', 0.0):.6f} | "
                f"{row['val_mae']:.6f} +- {row.get('val_mae_std', 0.0):.6f} | "
                f"{row.get('val_direction_acc', 0.0):.4f} +- {row.get('val_direction_acc_std', 0.0):.4f} | "
                f"{row['test_rmse']:.6f} +- {row.get('test_rmse_std', 0.0):.6f} | "
                f"{row['test_mae']:.6f} +- {row.get('test_mae_std', 0.0):.6f} | "
                f"{row.get('test_direction_acc', 0.0):.4f} +- {row.get('test_direction_acc_std', 0.0):.4f} | "
                f"{row.get('simplicity', 'unknown')} | {row.get('explainability', 'unknown')} |"
            )
        lines.extend(
            [
                "",
                "## Conclusions",
                f"- Current best: `{best_result['method_id']}`",
                f"  - Stable score: `{best_result.get('stable_score', self._stable_score(best_result)):.6f}`",
                f"  - Validation RMSE: `{best_result['val_rmse']:.6f} +- {best_result.get('val_rmse_std', 0.0):.6f}`",
                f"  - Test RMSE: `{best_result['test_rmse']:.6f} +- {best_result.get('test_rmse_std', 0.0):.6f}`",
                f"  - Representative seed: `{best_result.get('representative_seed', 'n/a')}`",
                f"- Generalization best: `{generalization_best['method_id']}`",
                f"  - Test RMSE: `{generalization_best['test_rmse']:.6f} +- {generalization_best.get('test_rmse_std', 0.0):.6f}`",
                f"  - Test MAE: `{generalization_best['test_mae']:.6f} +- {generalization_best.get('test_mae_std', 0.0):.6f}`",
                f"- Research mainline: `{research_mainline['method_id']}`",
                f"  - Stage: `{research_mainline['stage']}`",
                f"  - Explainability: `{research_mainline.get('explainability', 'unknown')}`",
                f"  - Representative seed: `{research_mainline.get('representative_seed', 'n/a')}`",
                "",
                "## Recommended Canonical Output",
                f"- Canonical fusion method: `{best_result['method_id']}`",
                f"- Artifact directory: `{best_result['method_dir']}`",
            ]
        )
        work_record_name = self.output_config.get("work_record_md")
        if work_record_name:
            with open(self.output_dir / work_record_name, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

    def _sync_best_artifacts(self, best_result: Dict[str, object]) -> None:
        method_dir = Path(best_result["method_dir"])
        for output_name in [
            self.output_config["features_file"],
            self.output_config["processed_file"],
            self.output_config["labels_file"],
            self.output_config["months_file"],
            self.output_config["missing_flags_file"],
            self.output_config["scaler_file"],
        ]:
            shutil.copy2(method_dir / output_name, self.output_dir / output_name)

    def run_single_method(
        self,
        stage: str,
        method: str,
        window_length: int,
        fused_dim: int,
        alpha: float,
        sync_canonical: bool = True,
    ) -> Dict[str, object]:
        prepared = self.prepare_inputs(window_length)
        result = self._train_method_multi_seed(stage, method, prepared, alpha, window_length, fused_dim)
        if sync_canonical:
            self._sync_best_artifacts(result)
            self._write_selection_outputs([result], result, prepared, window_length, fused_dim)
        return result

    def select_best(
        self,
        window_length: int,
        fused_dim: int,
        alpha: float,
        candidates: List[Tuple[str, str]] | None = None,
    ) -> Dict[str, object]:
        prepared = self.prepare_inputs(window_length)
        candidate_methods = candidates or FUSION_CONFIG["candidate_methods"]
        results = [
            self._train_method_multi_seed(stage, method, prepared, alpha, window_length, fused_dim)
            for stage, method in candidate_methods
        ]
        sorted_results = self._sort_results(results)
        best_result = sorted_results[0]
        self._sync_best_artifacts(best_result)
        self._write_selection_outputs(sorted_results, best_result, prepared, window_length, fused_dim)
        return best_result


