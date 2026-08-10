"""Small custom transformer encoder over traffic-flow feature tokens.

Design (compact on purpose -- runs in seconds on CPU):
  * Each of the ``n_features`` feature dimensions of a flow is treated as one
    token: its scalar value is passed through a learned ``Linear(1 -> d_model)``
    projection and added to a learned positional embedding per feature index.
  * An optional [CLS] token is prepended; its output vector is the sample
    embedding used for fine-tuning / t-SNE.
  * A learnable [MASK] embedding replaces masked feature tokens during
    pretraining; a regression head predicts the raw (standardized) feature
    value back, i.e. masked-feature reconstruction.
  * The stack is a small ``torch.nn.TransformerEncoder`` (2 layers, 2 heads,
    d_model=64). No ``transformers`` library needed.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        cls_token: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.cls_token = cls_token

        self.value_embed = nn.Linear(1, d_model)
        n_pos = n_features + (1 if cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(n_pos, d_model))
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        if cls_token:
            self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_embed, std=0.02)
        if cls_token:
            nn.init.normal_(self.cls_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim or (4 * d_model),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        self.reg_head = nn.Linear(d_model, 1)
        self.head_norm = nn.LayerNorm(d_model)

    def _forward_tokens(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """x: (B, F) float features; mask: (B, F) bool -- True means masked."""
        B, F = x.shape
        tokens = self.value_embed(x.unsqueeze(-1))  # (B, F, D)
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_embed, tokens)
        tokens = tokens + self.pos_embed[:F].unsqueeze(0)
        if self.cls_token:
            cls = self.cls_embed.expand(B, 1, -1) + self.pos_embed[F:].unsqueeze(0)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+F, D)
        return tokens

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Full token-sequence output (B, 1+F, D) for all tokens incl. [CLS]."""
        return self.encoder(self._forward_tokens(x, mask))

    def sample_embedding(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Sample-level embedding: mean pool over tokens (default) or [CLS]."""
        tokens = self.forward(x)
        return self.sample_embedding_from_tokens(tokens, pool=pool)

    def sample_embedding_from_tokens(self, tokens: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Pool an already-computed token sequence into a sample embedding."""
        if pool == "cls":
            return tokens[:, 0]
        return tokens.mean(dim=1)

    def cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Back-compat alias -- [CLS] token embedding."""
        return self.sample_embedding(x, pool="cls")

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict raw feature values for the masked positions -> (B, F)."""
        out = self.forward(x, mask)
        preds = self.reg_head(self.head_norm(out[:, 1:])).squeeze(-1)
        return preds


def build_encoder(
    n_features: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> FeatureTransformer:
    return FeatureTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
