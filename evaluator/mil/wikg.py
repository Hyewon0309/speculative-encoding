"""WiKG — Whole-slide Image Knowledge Graph MIL (vendored from MIL-Lab).

Uses top-k sparse attention to build a knowledge graph over patches,
then aggregates with bi-interaction and attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Classifier_1fc, DimReduction
from .utils import initialize_weights


class _AttnPool(nn.Module):
    """Simple attention-based global pooling (no torch_geometric dependency)."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, N, D) or (N, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        A = self.gate(x)                     # (B, N, 1)
        A = F.softmax(A, dim=1)              # (B, N, 1)
        out = (A * x).sum(dim=1)             # (B, D)
        return out


class WiKG(nn.Module):
    """
    Simplified WiKG-MIL: patch embed → sparse KG attention → bi-interaction → pool → classify.
    """

    def __init__(self, conf, topk=6, agg_type='bi-interaction', dropout=0.25):
        super().__init__()
        D_feat = conf.D_feat
        D_inner = conf.D_inner
        n_class = conf.n_class

        self.topk = topk
        self.agg_type = agg_type

        # patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(D_feat, D_inner),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # attention head/tail projections
        self.W_head = nn.Linear(D_inner, D_inner)
        self.W_tail = nn.Linear(D_inner, D_inner)
        self.scale = D_inner ** -0.5

        # gating
        self.gate_U = nn.Linear(D_inner, D_inner // 2)
        self.gate_V = nn.Linear(D_inner, D_inner // 2)
        self.gate_W = nn.Linear(D_inner // 2, D_inner)

        # aggregation
        if agg_type == 'gcn':
            self.linear = nn.Linear(D_inner, D_inner)
        elif agg_type == 'sage':
            self.linear = nn.Linear(D_inner * 2, D_inner)
        elif agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(D_inner, D_inner)
            self.linear2 = nn.Linear(D_inner, D_inner)

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm = nn.LayerNorm(D_inner)
        self.pool = _AttnPool(D_inner)
        self.classifier = nn.Linear(D_inner, n_class)

        initialize_weights(self)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, D) → (1, N, D)

        h = self.patch_embed(x)
        # mean-shift
        h = (h + h.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(h)  # (B, N, D)
        e_t = self.W_tail(h)  # (B, N, D)

        # sparse attention: top-k neighbors
        attn_logit = (e_h @ e_t.transpose(-2, -1)) * self.scale  # (B, N, N)
        k = min(self.topk, attn_logit.shape[-1])
        topk_weight, topk_index = torch.topk(attn_logit, k=k, dim=-1)  # (B, N, k)

        # gather neighbor features
        batch_idx = torch.arange(e_t.size(0), device=h.device).view(-1, 1, 1)
        Nb_h = e_t[batch_idx, topk_index]  # (B, N, k, D)

        topk_prob = F.softmax(topk_weight, dim=-1)  # (B, N, k)
        eh_r = topk_prob.unsqueeze(-1) * Nb_h  # (B, N, k, D)

        # knowledge-aware attention
        gate = torch.tanh(e_h.unsqueeze(2).expand_as(Nb_h) + eh_r)
        ka_weight = (gate * Nb_h).sum(dim=-1)  # (B, N, k)
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(-1)  # (B, N, k, 1)
        e_Nh = (ka_prob * Nb_h).sum(dim=2)  # (B, N, D)

        # aggregation
        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear(e_h + e_Nh))
        elif self.agg_type == 'sage':
            embedding = self.activation(self.linear(torch.cat([e_h, e_Nh], dim=-1)))
        elif self.agg_type == 'bi-interaction':
            sum_emb = self.activation(self.linear1(e_h + e_Nh))
            bi_emb = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_emb + bi_emb

        embedding = self.message_dropout(embedding)
        h_pool = self.pool(embedding)  # (B, D)
        h_norm = self.norm(h_pool)
        return self.classifier(h_norm)
