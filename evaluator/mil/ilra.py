"""ILRA (from PathGen architecture.ilra)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import initialize_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU()) if gated else None

    def forward(self, Q, K):
        Q0 = Q
        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)
        A, _ = self.multihead_attn(Q, K, V)
        O = (Q + A).transpose(0, 1)
        if self.ln0 is not None:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.ln1 is not None:
            O = self.ln1(O)
        if self.gate is not None:
            O = O * self.gate(Q0)
        return O


class GAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.latent)
        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)
        X_hat = self.project_backward(X, H)
        return X_hat


class NLP(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        return self.mha(global_embedding, X)


class ILRA(nn.Module):
    def __init__(self, num_layers=2, feat_dim=768, n_classes=2, hidden_feat=256, num_heads=8, topk=1, ln=False):
        super().__init__()
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(
                dim_in=feat_dim if idx == 0 else hidden_feat,
                dim_out=hidden_feat,
                num_heads=num_heads,
                num_inds=topk,
                ln=ln,
            )
            gab_blocks.append(block)
        self.gab_blocks = nn.ModuleList(gab_blocks)
        self.pooling = NLP(dim=hidden_feat, num_heads=num_heads, num_seeds=topk, ln=ln)
        self.classifier = nn.Linear(hidden_feat, n_classes)
        initialize_weights(self)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        for block in self.gab_blocks:
            x = block(x)
        feat = self.pooling(x)
        logits = self.classifier(feat).squeeze(1)
        return logits
