from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from fusion.config import FUSION_CONFIG


def _make_projection(input_dim: int, output_dim: int) -> nn.Module:
    if input_dim == output_dim:
        return nn.Identity()
    return nn.Linear(input_dim, output_dim)


class BaseFusionModule(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__()
        self.text_proj = _make_projection(text_dim, fused_dim)
        self.image_proj = _make_projection(image_dim, fused_dim)
        self.context_dim = context_dim
        self.context_proj = _make_projection(context_dim, fused_dim) if context_dim > 0 else None
        self.fused_dim = fused_dim

    def project_modalities(
        self, text_feat: torch.Tensor, image_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_proj(text_feat), self.image_proj(image_feat)

    def project_context(
        self,
        context_feat: torch.Tensor | None,
        reference_feat: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.context_dim <= 0:
            return None
        if context_feat is None:
            return torch.zeros_like(reference_feat)
        return self.context_proj(context_feat)

    def _package_outputs(
        self,
        fused_feat: torch.Tensor,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor,
        return_aux: bool,
        extra: Dict[str, torch.Tensor] | None = None,
    ):
        if not return_aux:
            return fused_feat
        output = {
            "fused_feat": fused_feat,
            "text_proj": text_proj,
            "image_proj": image_proj,
        }
        if extra:
            output.update(extra)
        return output


class EarlyConcatFusion(BaseFusionModule):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        self.output_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_proj, image_proj = self.project_modalities(text_feat, image_feat)
        fused_feat = self.output_proj(torch.cat([text_proj, image_proj], dim=-1))
        return self._package_outputs(fused_feat, text_proj, image_proj, return_aux)


class WeightedSumFusion(BaseFusionModule):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fused_dim: int,
        context_dim: int = 0,
        alpha: float = 0.5,
        learnable: bool = False,
    ) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        self.learnable = learnable
        clipped_alpha = min(max(alpha, 1e-4), 1 - 1e-4)
        alpha_raw = torch.logit(torch.tensor(clipped_alpha, dtype=torch.float32))
        if learnable:
            self.alpha_raw = nn.Parameter(alpha_raw.clone())
        else:
            self.register_buffer("alpha_buffer", torch.tensor(alpha, dtype=torch.float32))

    def get_alpha(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.alpha_raw)
        return self.alpha_buffer

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_proj, image_proj = self.project_modalities(text_feat, image_feat)
        alpha = self.get_alpha().to(text_proj.device)
        fused_feat = alpha * text_proj + (1.0 - alpha) * image_proj
        return self._package_outputs(
            fused_feat,
            text_proj,
            image_proj,
            return_aux,
            extra={"alpha": alpha.detach()},
        )


class GatedFusion(BaseFusionModule):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        gate_input_dim = fused_dim * (3 if context_dim > 0 else 2)
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_proj, image_proj = self.project_modalities(text_feat, image_feat)
        context_proj = self.project_context(context_feat, text_proj)
        gate_inputs = [text_proj, image_proj]
        if context_proj is not None:
            gate_inputs.append(context_proj)
        gate = torch.sigmoid(self.gate_proj(torch.cat(gate_inputs, dim=-1)))
        fused_feat = gate * text_proj + (1.0 - gate) * image_proj
        return self._package_outputs(
            fused_feat,
            text_proj,
            image_proj,
            return_aux,
            extra={"gate": gate, "context_proj": context_proj},
        )


class TemporalContextAttentionFusion(BaseFusionModule):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        score_input_dim = fused_dim * (3 if context_dim > 0 else 2)
        self.score_proj = nn.Sequential(
            nn.Linear(score_input_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(FUSION_CONFIG["dropout"]),
            nn.Linear(fused_dim, 2),
        )
        self.context_residual = nn.Linear(fused_dim, fused_dim) if context_dim > 0 else None
        self.output_norm = nn.LayerNorm(fused_dim)

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_proj, image_proj = self.project_modalities(text_feat, image_feat)
        context_proj = self.project_context(context_feat, text_proj)
        score_inputs = [text_proj, image_proj]
        if context_proj is not None:
            score_inputs.append(context_proj)
        modality_logits = self.score_proj(torch.cat(score_inputs, dim=-1))
        modality_weights = torch.softmax(modality_logits, dim=-1)
        fused_feat = (
            modality_weights[..., 0:1] * text_proj
            + modality_weights[..., 1:2] * image_proj
        )
        if context_proj is not None and self.context_residual is not None:
            fused_feat = fused_feat + self.context_residual(context_proj)
        fused_feat = self.output_norm(fused_feat)
        return self._package_outputs(
            fused_feat,
            text_proj,
            image_proj,
            return_aux,
            extra={
                "context_proj": context_proj,
                "modality_weights": modality_weights,
            },
        )


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, fused_dim: int) -> None:
        super().__init__()
        self.input_proj = _make_projection(input_dim, fused_dim)
        self.encoder = nn.GRU(
            input_size=fused_dim,
            hidden_size=fused_dim,
            num_layers=FUSION_CONFIG["gru_layers"],
            batch_first=True,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feat)
        out, _ = self.encoder(x)
        return out


class BaseLateFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__()
        self.text_encoder = GRUEncoder(text_dim, fused_dim)
        self.image_encoder = GRUEncoder(image_dim, fused_dim)

    def encode_modalities(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        return self.text_encoder(text_feat), self.image_encoder(image_feat)

    def _package_outputs(
        self,
        fused_feat: torch.Tensor,
        text_out: torch.Tensor,
        image_out: torch.Tensor,
        return_aux: bool,
        extra: Dict[str, torch.Tensor] | None = None,
    ):
        if not return_aux:
            return fused_feat
        output = {
            "fused_feat": fused_feat,
            "text_proj": text_out,
            "image_proj": image_out,
            "text_out": text_out,
            "image_out": image_out,
        }
        if extra:
            output.update(extra)
        return output


class LateConcatFusion(BaseLateFusion):
    def __init__(self, text_dim: int, image_dim: int, fused_dim: int, context_dim: int = 0) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        self.output_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_out, image_out = self.encode_modalities(text_feat, image_feat)
        fused_feat = self.output_proj(torch.cat([text_out, image_out], dim=-1))
        return self._package_outputs(fused_feat, text_out, image_out, return_aux)


class LateWeightedFusion(BaseLateFusion):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fused_dim: int,
        context_dim: int = 0,
        alpha: float = 0.5,
        learnable: bool = True,
    ) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        self.learnable = learnable
        clipped_alpha = min(max(alpha, 1e-4), 1 - 1e-4)
        alpha_raw = torch.logit(torch.tensor(clipped_alpha, dtype=torch.float32))
        if learnable:
            self.alpha_raw = nn.Parameter(alpha_raw.clone())
        else:
            self.register_buffer("alpha_buffer", torch.tensor(alpha, dtype=torch.float32))

    def get_alpha(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.alpha_raw)
        return self.alpha_buffer

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_out, image_out = self.encode_modalities(text_feat, image_feat)
        alpha = self.get_alpha().to(text_out.device)
        fused_feat = alpha * text_out + (1.0 - alpha) * image_out
        return self._package_outputs(
            fused_feat,
            text_out,
            image_out,
            return_aux,
            extra={"alpha": alpha.detach()},
        )


class LateGateFusion(BaseLateFusion):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fused_dim: int,
        context_dim: int = 0,
        gate_mode: str = "context_mlp",
        gate_bias_init: float = -1.0,
        gate_regularization_lambda: float = 0.02,
        gate_max_text_share: float = 0.45,
        gate_residual_scale: float = 0.15,
        concat_preserve_scale: float = 0.0,
        text_modal_dropout: float = 0.20,
        image_modal_dropout: float = 0.05,
    ) -> None:
        super().__init__(text_dim, image_dim, fused_dim, context_dim=context_dim)
        self.gate_mode = gate_mode
        self.gate_regularization_lambda = float(gate_regularization_lambda)
        self.gate_max_text_share = float(gate_max_text_share)
        self.gate_residual_scale = float(gate_residual_scale)
        self.concat_preserve_scale = float(concat_preserve_scale)
        self.text_modal_dropout = float(text_modal_dropout)
        self.image_modal_dropout = float(image_modal_dropout)
        self.context_proj = nn.Linear(context_dim, fused_dim) if context_dim > 0 and gate_mode == "context_mlp" else None
        gate_input_dim = fused_dim * (3 if self.context_proj is not None else 2)
        if gate_mode == "simple":
            self.gate_proj = nn.Linear(fused_dim * 2, fused_dim)
            final_layer = self.gate_proj
        elif gate_mode == "context_mlp":
            self.gate_proj = nn.Sequential(
                nn.Linear(gate_input_dim, fused_dim),
                nn.GELU(),
                nn.Dropout(FUSION_CONFIG["dropout"]),
                nn.Linear(fused_dim, fused_dim),
            )
            final_layer = self.gate_proj[-1]
        else:
            raise ValueError(f"Unsupported late gate mode: {gate_mode}")
        nn.init.constant_(final_layer.bias, float(gate_bias_init))
        self.residual_proj = nn.Linear(gate_input_dim, fused_dim) if self.gate_residual_scale > 0 else None
        self.concat_preserve_proj = (
            nn.Sequential(nn.LayerNorm(gate_input_dim), nn.Linear(gate_input_dim, fused_dim))
            if self.concat_preserve_scale > 0
            else None
        )
        self.output_norm = (
            nn.LayerNorm(fused_dim)
            if self.gate_residual_scale > 0 or self.concat_preserve_scale > 0
            else nn.Identity()
        )

    def _modal_dropout(self, feat: torch.Tensor, probability: float) -> torch.Tensor:
        if not self.training or probability <= 0:
            return feat
        keep_prob = 1.0 - probability
        if keep_prob <= 0:
            return torch.zeros_like(feat)
        mask = torch.bernoulli(torch.full((feat.shape[0], 1, 1), keep_prob, device=feat.device, dtype=feat.dtype))
        return feat * mask / keep_prob

    def auxiliary_loss(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.gate_regularization_lambda <= 0 or "gate" not in aux:
            gate = aux.get("gate")
            if isinstance(gate, torch.Tensor):
                return gate.new_tensor(0.0)
            return torch.tensor(0.0)
        gate = aux["gate"]
        text_share = gate.mean()
        penalty = torch.relu(text_share - self.gate_max_text_share) ** 2
        return penalty * self.gate_regularization_lambda

    def forward(self, text_feat, image_feat, context_feat=None, return_aux: bool = False):
        text_out, image_out = self.encode_modalities(text_feat, image_feat)
        dropped_text = self._modal_dropout(text_out, self.text_modal_dropout)
        dropped_image = self._modal_dropout(image_out, self.image_modal_dropout)
        gate_inputs = [dropped_text, dropped_image]
        context_out = None
        if self.context_proj is not None and context_feat is not None:
            context_out = self.context_proj(context_feat)
            gate_inputs.append(context_out)
        gate_input = torch.cat(gate_inputs, dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        fused_feat = gate * dropped_text + (1.0 - gate) * dropped_image
        if self.concat_preserve_proj is not None:
            fused_feat = fused_feat + self.concat_preserve_scale * self.concat_preserve_proj(gate_input)
        if self.residual_proj is not None:
            fused_feat = fused_feat + self.gate_residual_scale * torch.tanh(self.residual_proj(gate_input))
        fused_feat = self.output_norm(fused_feat)
        return self._package_outputs(
            fused_feat,
            text_out,
            image_out,
            return_aux,
            extra={"gate": gate, "context_proj": context_out},
        )


FUSION_REGISTRY = {
    ("early", "concat"): EarlyConcatFusion,
    ("early", "weighted_sum_fixed"): WeightedSumFusion,
    ("early", "weighted_sum_learnable"): WeightedSumFusion,
    ("intermediate", "gated_fusion"): GatedFusion,
    ("intermediate", "temporal_attention_fusion"): TemporalContextAttentionFusion,
    ("late", "gru_concat"): LateConcatFusion,
    ("late", "gru_weighted"): LateWeightedFusion,
    ("late", "gru_gate"): LateGateFusion,
    ("late", "gru_gate_baseline"): LateGateFusion,
    ("late", "gru_gate_context"): LateGateFusion,
    ("late", "gru_gate_context_image_biased"): LateGateFusion,
    ("late", "gru_gate_context_image_biased_regularized"): LateGateFusion,
    ("late", "gru_gate_concat_preserving"): LateGateFusion,
}


def build_fusion_module(
    stage: str,
    method: str,
    text_dim: int,
    image_dim: int,
    fused_dim: int,
    context_dim: int = 0,
    alpha: float = 0.5,
) -> nn.Module:
    key = (stage, method)
    if key not in FUSION_REGISTRY:
        raise ValueError(f"Unsupported fusion method: {stage}.{method}")

    module_cls = FUSION_REGISTRY[key]
    if key == ("early", "weighted_sum_fixed"):
        return module_cls(text_dim, image_dim, fused_dim, context_dim=context_dim, alpha=alpha, learnable=False)
    if key == ("early", "weighted_sum_learnable"):
        return module_cls(text_dim, image_dim, fused_dim, context_dim=context_dim, alpha=alpha, learnable=True)
    if key == ("late", "gru_weighted"):
        return module_cls(text_dim, image_dim, fused_dim, context_dim=context_dim, alpha=alpha, learnable=True)
    if stage == "late" and method.startswith("gru_gate"):
        variant_config = dict(FUSION_CONFIG["late_gate_variants"].get(method, FUSION_CONFIG["late_gate_variants"]["gru_gate"]))
        if method == "gru_gate":
            for option_name in (
                "gate_mode",
                "gate_bias_init",
                "gate_regularization_lambda",
                "gate_max_text_share",
                "gate_residual_scale",
                "concat_preserve_scale",
                "text_modal_dropout",
                "image_modal_dropout",
            ):
                if option_name in FUSION_CONFIG:
                    variant_config[option_name] = FUSION_CONFIG[option_name]
        return module_cls(
            text_dim,
            image_dim,
            fused_dim,
            context_dim=context_dim,
            **variant_config,
        )
    return module_cls(text_dim, image_dim, fused_dim, context_dim=context_dim)
