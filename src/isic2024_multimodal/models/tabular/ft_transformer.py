from __future__ import annotations

from typing import Any


class FTTransformerBinaryModel:
    """Compact FT-Transformer style binary classifier for dense tabular matrices."""

    def __init__(
        self,
        input_dim: int,
        *,
        d_token: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ) -> None:
        import torch
        import torch.nn as nn

        self.input_dim = int(input_dim)
        self.d_token = int(d_token)
        self.module = nn.Module()
        self.module.feature_weight = nn.Parameter(torch.empty(self.input_dim, self.d_token))
        self.module.feature_bias = nn.Parameter(torch.zeros(self.input_dim, self.d_token))
        self.module.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_token,
            nhead=int(n_heads),
            dim_feedforward=self.d_token * 4,
            dropout=float(ffn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.module.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_blocks))
        self.module.norm = nn.LayerNorm(self.d_token)
        self.module.head = nn.Sequential(
            nn.Dropout(float(attention_dropout)),
            nn.Linear(self.d_token, 1),
        )
        nn.init.xavier_uniform_(self.module.feature_weight)
        nn.init.normal_(self.module.cls_token, std=0.02)

    def __call__(self, x):
        import torch

        tokens = x.unsqueeze(-1) * self.module.feature_weight.unsqueeze(0) + self.module.feature_bias.unsqueeze(0)
        cls = self.module.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.module.encoder(torch.cat([cls, tokens], dim=1))
        cls_encoded = self.module.norm(encoded[:, 0])
        return self.module.head(cls_encoded).squeeze(1)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def to(self, device):
        self.module.to(device)
        return self


def build_external_ft_transformer_binary_model(input_dim: int, hyperparameters: dict[str, Any]):
    try:
        import rtdl_revisiting_models  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ft_transformer_external requires the optional package `rtdl_revisiting_models`. "
            "Install dependencies from requirements.txt or run the repo-native `ft_transformer` baseline instead."
        ) from exc

    # The external package has changed APIs across releases. Keep the public model
    # name available, but route construction through the repo-native compatible
    # module until the exact external API is pinned by an experiment config.
    return FTTransformerBinaryModel(
        input_dim,
        d_token=int(hyperparameters.get("d_token", 64)),
        n_blocks=int(hyperparameters.get("n_blocks", 2)),
        n_heads=int(hyperparameters.get("n_heads", 4)),
        attention_dropout=float(hyperparameters.get("attention_dropout", 0.1)),
        ffn_dropout=float(hyperparameters.get("ffn_dropout", 0.1)),
    )
