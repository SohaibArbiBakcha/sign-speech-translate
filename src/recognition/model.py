"""Transformer encoder over per-frame pose/hand keypoints for isolated sign
(gloss) classification."""
import torch
import torch.nn as nn


class SignTransformer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, max_frames: int = 96, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
