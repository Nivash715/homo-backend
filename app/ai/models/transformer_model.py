"""
Transformer Model for advanced threat intelligence analysis and cyber attack prediction.
Uses PyTorch with multi-head self-attention.
"""
from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CyberTransformer(nn.Module):
    """
    Transformer for classifying threat intelligence records.
    Input: (batch, seq_len, feature_dim) — tokenised threat feature vectors.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over sequence
        return self.classifier(x)


def build_transformer_model(
    feature_dim: int = 64, num_classes: int = 2
) -> "CyberTransformer":
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    return CyberTransformer(feature_dim=feature_dim, num_classes=num_classes)


def get_torch_weights(model: "CyberTransformer") -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_torch_weights(model: "CyberTransformer", weights: list[np.ndarray]) -> None:
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.tensor(w))


def train_transformer_step(model, optimizer, X_batch, y_batch) -> float:
    import torch.nn.functional as F
    model.train()
    optimizer.zero_grad()
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()
