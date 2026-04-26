from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fusion.config import FUSION_CONFIG, timemixer_freq
from fusion.modules import build_fusion_module


TIME_MIXER_ROOT = Path(__file__).resolve().parents[2] / "3_modeling" / "TimeMixer"
if str(TIME_MIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIME_MIXER_ROOT))

from models.TimeMixer import Model as TimeMixerModel


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
    device: torch.device,
    pred_len: int = 1,
    frequency: str = "monthly",
) -> SimpleNamespace:
    down_sampling_window = FUSION_CONFIG["timemixer_down_sampling_window"]
    down_sampling_layers = resolve_down_sampling_layers(
        seq_len=seq_len,
        window=down_sampling_window,
        requested_layers=FUSION_CONFIG["timemixer_down_sampling_layers"],
    )
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=0,
        pred_len=pred_len,
        down_sampling_window=down_sampling_window,
        down_sampling_layers=down_sampling_layers,
        down_sampling_method=FUSION_CONFIG["timemixer_down_sampling_method"],
        channel_independence=FUSION_CONFIG["timemixer_channel_independence"],
        decomp_method=FUSION_CONFIG["timemixer_decomp_method"],
        moving_avg=FUSION_CONFIG["timemixer_moving_avg"],
        top_k=FUSION_CONFIG["timemixer_top_k"],
        d_model=FUSION_CONFIG["timemixer_d_model"],
        n_heads=FUSION_CONFIG["timemixer_n_heads"],
        e_layers=FUSION_CONFIG["timemixer_e_layers"],
        d_ff=FUSION_CONFIG["timemixer_d_ff"],
        dropout=FUSION_CONFIG["timemixer_dropout"],
        use_norm=FUSION_CONFIG["timemixer_use_norm"],
        embed=FUSION_CONFIG["timemixer_embed"],
        freq=timemixer_freq(frequency),
        use_future_temporal_feature=FUSION_CONFIG["timemixer_use_future_temporal_feature"],
        enc_in=input_dim,
        dec_in=1,
        c_out=1,
        output_attention=False,
        model="TimeMixer",
        use_gpu=device.type == "cuda",
        use_multi_gpu=False,
        gpu=0,
        devices="0",
    )


class FusionTimeMixerForecaster(nn.Module):
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
        seq_len: int,
        frequency: str = "monthly",
    ) -> None:
        super().__init__()
        self.pred_len = 1
        self.frequency = frequency
        self.time_mark_dim = 3 if frequency == "daily" else 1
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
        self.final_dim = fused_dim * 2
        self.timemixer_args = build_timemixer_args(
            seq_len=seq_len,
            input_dim=self.final_dim,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            pred_len=self.pred_len,
            frequency=frequency,
        )
        self.timemixer = TimeMixerModel(self.timemixer_args).float()

    def auxiliary_loss(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.fusion_module, "auxiliary_loss"):
            return self.fusion_module.auxiliary_loss(aux)
        return aux["final_seq"].new_tensor(0.0)

    def encode_sequence(
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
        if not return_aux:
            return final_seq

        output = {
            "structured_proj": structured_proj,
            "fused_text_image": fused_text_image,
            "final_seq": final_seq,
        }
        output.update(fusion_out)
        return output

    def forward(
        self,
        structured_feat: torch.Tensor,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor,
        context_feat: torch.Tensor,
        return_aux: bool = False,
    ):
        fusion_bundle = self.encode_sequence(structured_feat, text_feat, image_feat, context_feat, return_aux=True)
        final_seq = fusion_bundle["final_seq"]
        batch_size, seq_len, _ = final_seq.shape
        x_mark = torch.zeros(batch_size, seq_len, self.time_mark_dim, device=final_seq.device, dtype=final_seq.dtype)
        y_mark = torch.zeros(
            batch_size,
            self.pred_len,
            self.time_mark_dim,
            device=final_seq.device,
            dtype=final_seq.dtype,
        )
        dec_inp = None
        outputs = self.timemixer(final_seq, x_mark, dec_inp, y_mark)
        pred = outputs[:, -1, 0]

        if not return_aux:
            return pred

        fusion_bundle["prediction"] = pred
        return fusion_bundle


class TimeMixerSelectorTrainer:
    def __init__(self, device: torch.device, target_mode: str = "level") -> None:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.device = device
        self.batch_size = FUSION_CONFIG["timemixer_batch_size"]
        self.max_epochs = FUSION_CONFIG["timemixer_max_epochs"]
        self.patience = FUSION_CONFIG["timemixer_patience"]
        self.learning_rate = FUSION_CONFIG["timemixer_learning_rate"]
        self.weight_decay = FUSION_CONFIG["timemixer_weight_decay"]
        self.target_mode = target_mode

    def _make_loader(self, split_dict: Dict[str, np.ndarray], shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(split_dict["structured"], dtype=torch.float32),
            torch.tensor(split_dict["text"], dtype=torch.float32),
            torch.tensor(split_dict["image"], dtype=torch.float32),
            torch.tensor(split_dict["context"], dtype=torch.float32),
            torch.tensor(split_dict["labels"], dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _metrics(self, pred: np.ndarray, true: np.ndarray, reference: np.ndarray | None = None) -> Dict[str, float]:
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

    def evaluate_split(self, model: FusionTimeMixerForecaster, split_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        model.eval()
        loader = self._make_loader(split_dict, shuffle=False)
        preds = []
        trues = []
        with torch.no_grad():
            for structured_feat, text_feat, image_feat, context_feat, labels in loader:
                structured_feat = structured_feat.to(self.device, non_blocking=True)
                text_feat = text_feat.to(self.device, non_blocking=True)
                image_feat = image_feat.to(self.device, non_blocking=True)
                context_feat = context_feat.to(self.device, non_blocking=True)
                pred = model(structured_feat, text_feat, image_feat, context_feat).detach().cpu().numpy()
                preds.append(pred)
                trues.append(labels.numpy())

        pred_array = np.concatenate(preds, axis=0)
        true_array = np.concatenate(trues, axis=0)
        reference = split_dict["reference"]
        return self._metrics(pred_array, true_array, reference)

    def fit(
        self,
        model: FusionTimeMixerForecaster,
        train_split: Dict[str, np.ndarray],
        valid_split: Dict[str, np.ndarray],
        test_split: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, object]:
        train_loader = self._make_loader(train_split, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        best_state = deepcopy(model.state_dict())
        best_val_rmse = math.inf
        best_epoch = 0
        best_val_metrics = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            model.train()
            for structured_feat, text_feat, image_feat, context_feat, labels in train_loader:
                structured_feat = structured_feat.to(self.device, non_blocking=True)
                text_feat = text_feat.to(self.device, non_blocking=True)
                image_feat = image_feat.to(self.device, non_blocking=True)
                context_feat = context_feat.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = model(structured_feat, text_feat, image_feat, context_feat, return_aux=True)
                pred = output["prediction"]
                loss = loss_fn(pred, labels) + model.auxiliary_loss(output)
                loss.backward()
                optimizer.step()

            val_metrics = self.evaluate_split(model, valid_split)
            if val_metrics["rmse"] < best_val_rmse - 1e-12:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        model.load_state_dict(best_state)
        train_metrics = self.evaluate_split(model, train_split)
        val_metrics = best_val_metrics or self.evaluate_split(model, valid_split)
        test_metrics = self.evaluate_split(model, test_split) if test_split is not None else None
        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "best_epoch": best_epoch,
            "backend": "timemixer",
            "timemixer_config": {
                "d_model": model.timemixer_args.d_model,
                "n_heads": model.timemixer_args.n_heads,
                "e_layers": model.timemixer_args.e_layers,
                "d_ff": model.timemixer_args.d_ff,
                "down_sampling_layers": model.timemixer_args.down_sampling_layers,
                "down_sampling_window": model.timemixer_args.down_sampling_window,
                "channel_independence": model.timemixer_args.channel_independence,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
            },
        }
