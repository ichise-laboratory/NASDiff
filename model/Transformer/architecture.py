"""
Vanilla Transformer baseline adapted for the repository imputation protocol.

The backbone follows a standard Transformer-imputation adaptation: concatenate
values with their missing mask, pass the sequence through Transformer encoder
layers, and reconstruct the original feature space.
"""
from typing import Callable

import torch
import torch.nn as nn
from pypots.nn.modules import ModelCore
from pypots.nn.modules.saits import SaitsEmbedding
from pypots.nn.modules.transformer import (
    ScaledDotProductAttention,
    TransformerEncoderLayer,
)
from pypots.nn.functional import calc_mae


class Transformer(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
        self.layer_stack = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.reduce_dim = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, stage: str = "train") -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        enc_output = self.embedding(X, missing_mask)
        attn_weights = None
        for encoder_layer in self.layer_stack:
            enc_output, attn_weights = encoder_layer(enc_output, None)

        reconstruction = self.reduce_dim(enc_output)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        return {
            "imputed_result": imputed_data,
            "reconstruction": reconstruction,
            "attn_weights": attn_weights,
        }

    def predict(self, inputs: dict, **kwargs) -> dict:
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs)

        return output

    def loss_func(
        self,
        outputs: dict,
        inputs: dict,
        calc_func: Callable = calc_mae,
    ) -> torch.Tensor:
        X = inputs["X"]
        X_intact = inputs["X_intact"]
        missing_mask = inputs["missing_mask"]
        indicating_mask = inputs["indicating_mask"]
        reconstruction = outputs["reconstruction"]

        ORT_loss = self.ORT_weight * calc_func(reconstruction, X, missing_mask)
        MIT_loss = self.MIT_weight * calc_func(reconstruction, X_intact, indicating_mask)

        return ORT_loss + MIT_loss
