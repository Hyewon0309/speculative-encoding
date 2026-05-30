"""RRT-MIL — Region-based Recurrent Transformer MIL (vendored from MIL-Lab).

Simplified version using NystromAttention + PPEG + attention pooling.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from .network import Classifier_1fc, DimReduction
from .nystrom_attention import NystromAttention
from .utils import initialize_weights


class PPEG(nn.Module):
    def __init__(self, dim=512, k=7):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        if add_length > 0:
            x = torch.cat([x, x[:, :add_length, :]], dim=1)
        if H < 7:
            _H = 7
            zero_pad = _H * _H - (N + add_length)
            x = torch.cat([x, torch.zeros(B, zero_pad, C, device=x.device)], dim=1)
            add_length += zero_pad
            H = W = _H

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]
        return x


class _Attention(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super().__init__()
        D = 128
        layers = [nn.Linear(input_dim, D)]
        if act == 'gelu':
            layers.append(nn.GELU())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, 1, bias=bias))
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, C) or (N, C)
        A = self.attention(x)      # (..., 1)
        A = A.transpose(-1, -2)    # (..., 1, N)
        A = F.softmax(A, dim=-1)
        out = torch.matmul(A, x)   # (..., 1, C)
        return out.squeeze(-2)     # (..., C)


class _TransLayer(nn.Module):
    def __init__(self, dim=512, n_heads=8, drop_out=0.1, drop_path=0.0,
                 ffn=False, mlp_ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // n_heads,
            heads=n_heads,
            num_landmarks=min(256, dim // 2),
            pinv_iterations=6,
            residual=True,
            dropout=drop_out,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = ffn
        if ffn:
            self.norm2 = nn.LayerNorm(dim)
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop_out),
                nn.Linear(hidden, dim), nn.Dropout(drop_out),
            )
        else:
            self.norm2 = nn.Identity()
            self.mlp = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RRT(nn.Module):
    """Simplified RRT-MIL: dim-reduce → Nystrom layers + PPEG → attn pool → classify."""

    def __init__(self, conf, n_layers=2, n_heads=8, drop_out=0.1, drop_path=0.1,
                 ffn=True, mlp_ratio=4.0, peg_k=7, dropout=0.25):
        super().__init__()
        D_feat = conf.D_feat
        D_inner = conf.D_inner
        n_class = conf.n_class

        self.patch_embed = nn.Sequential(nn.Linear(D_feat, D_inner), nn.ReLU())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pos_embed = PPEG(dim=D_inner, k=peg_k)

        layers = []
        for _ in range(n_layers):
            layers.append(_TransLayer(
                dim=D_inner, n_heads=n_heads, drop_out=drop_out,
                drop_path=drop_path, ffn=ffn, mlp_ratio=mlp_ratio,
            ))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(D_inner)
        self.pool = _Attention(D_inner, act='relu', dropout=False)
        self.classifier = Classifier_1fc(D_inner, n_class, 0.0)

        initialize_weights(self)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, D) → (1, N, D)

        x = self.patch_embed(x)
        x = self.dropout(x)

        # first layer
        x = self.layers[0](x)
        # positional encoding after first layer
        x = x + self.pos_embed(x)
        # remaining layers
        for layer in self.layers[1:]:
            x = layer(x)

        x = self.norm(x)
        x = self.pool(x)       # (B, D_inner)
        return self.classifier(x)
