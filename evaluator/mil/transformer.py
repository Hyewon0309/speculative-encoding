"""MHA and ABMIL (from PathGen architecture.transformer)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .network import Classifier_1fc, DimReduction


class MutiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, dropout: float = 0.1,
                 n_masked_patch: int = 0, mask_drop: float = 0.0):
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple:
        q = self._separate_heads(self.q_proj(q), self.num_heads)
        k = self._separate_heads(self.k_proj(k), self.num_heads)
        v = self._separate_heads(self.v_proj(v), self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(c_per_head)
        if self.n_masked_patch > 0 and self.training:
            b, h, qn, c = attn.shape
            n_mp = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_mp, dim=-1)
            indices = indices.reshape(b * h * qn, -1)
            rand_sel = torch.argsort(torch.rand(*indices.shape, device=indices.device), dim=-1)[:, :int(n_mp * self.mask_drop)]
            masked_idx = indices[torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1), rand_sel]
            random_mask = torch.ones(b * h * qn, c, device=attn.device)
            random_mask.scatter_(-1, masked_idx, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, qn, -1) == 0, -1e9)
        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        out1 = self._recombine_heads(attn @ v)
        out1 = self.layer_norm(self.dropout(self.out_proj(out1)))
        return out1[:, 0], attn_out[0]


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x):
        A = self.attention_weights(self.attention_V(x) * self.attention_U(x))
        return A.transpose(0, 1)


class MHA(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros(1, 1, conf.D_inner))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            input = input[0]
        if input.dim() == 3:
            input = input.squeeze(0)  # (1, N, D) -> (N, D) for single-bag batch
        x = self.dimreduction(input)       # (N, D_inner)
        x = x.unsqueeze(0)                 # (1, N, D_inner)
        q = self.q                          # (1, 1, D_inner)
        feat, _ = self.attention(q, x, x)  # (1, D_inner)
        return self.classifier(feat)


class ABMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 3:
            x = x.squeeze(0)  # (1, N, D) -> (N, D) for single-bag batch
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)
        A = F.softmax(A, dim=1)
        afeat = torch.mm(A, med_feat)
        return self.classifier(afeat)
