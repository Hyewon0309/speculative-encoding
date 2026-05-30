"""TransMIL (from PathGen architecture.transMIL)."""

import numpy as np
import torch
import torch.nn as nn

from .nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.pos_layer = PPEG(dim=conf.D_inner)
        self._fc1 = nn.Sequential(nn.Linear(conf.D_feat, conf.D_inner), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, conf.D_inner))
        self.n_classes = conf.n_class
        self.layer1 = TransLayer(dim=conf.D_inner)
        self.layer2 = TransLayer(dim=conf.D_inner)
        self.norm = nn.LayerNorm(conf.D_inner)
        self._fc2 = nn.Linear(conf.D_inner, conf.n_class)

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            input = input[0]
        h = self._fc1(input)
        H = h.shape[1]
        _H = _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        return self._fc2(h)
