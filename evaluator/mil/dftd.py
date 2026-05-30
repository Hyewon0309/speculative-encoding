"""DFTD — Double-Tier Feature Distillation MIL (vendored from MIL-Lab)."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Classifier_1fc, DimReduction
from .utils import initialize_weights


class _GatedAttention(nn.Module):
    def __init__(self, L=512, D=128, K=1, dropout=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, K)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # (N, K)
        A = F.softmax(A, dim=0)
        if A.dim() == 2:
            A = A.squeeze(-1)  # (N,)
        return A


class _AttentionWithClassifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super().__init__()
        self.attention = _GatedAttention(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        AA = self.attention(x)  # (N,)
        afeat = torch.sum(AA.unsqueeze(-1) * x, dim=0, keepdim=True)  # (1, D)
        pred = self.classifier(afeat)  # (1, num_cls)
        return pred, afeat, AA


class DFTD(nn.Module):
    """
    Simplified DFTD for vendored MIL.

    Training mode: forward(x, label) → (logits, instance_loss)
    Eval mode:     forward(x) → logits
    """

    def __init__(self, conf, distill='MaxMinS', num_group=8, total_instance=8,
                 bag_weight=0.7, dropout=0.25):
        super().__init__()
        D_feat = conf.D_feat
        D_inner = conf.D_inner
        n_class = conf.n_class

        self.dimreduction = DimReduction(D_feat, D_inner)
        self.attention = _GatedAttention(D_inner, 128, 1, dropout)
        self.classifier = Classifier_1fc(D_inner, n_class, dropout)
        self.attCls = _AttentionWithClassifier(D_inner, 128, 1, n_class, dropout)

        self.distill = distill
        self.num_group = num_group
        self.total_instance = total_instance
        self.bag_weight = bag_weight
        self.inst_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.n_class = n_class

        initialize_weights(self)

    def _get_cam_1d(self, features):
        tweight = list(self.classifier.parameters())[0]  # (n_class, D_inner)
        cam_maps = torch.einsum('nf,cf->nc', features, tweight)
        return cam_maps

    def forward(self, x, label=None):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 3:
            x = x.squeeze(0)  # (N, D)

        instance_per_group = max(1, self.total_instance // self.num_group)

        slide_pseudo_feat = []
        slide_sub_preds = []

        feat_index = list(range(x.shape[0]))
        random.shuffle(feat_index)
        index_chunks = np.array_split(np.array(feat_index), self.num_group)
        index_chunks = [c.tolist() for c in index_chunks]

        for tindex in index_chunks:
            sub_feat = x[tindex]  # (G, D_feat)
            mid_feat = self.dimreduction(sub_feat)  # (G, D_inner)

            tAA = self.attention(mid_feat)  # (G,)
            att_feats = mid_feat * tAA.unsqueeze(-1)  # weighted
            att_feat_sum = att_feats.sum(dim=0, keepdim=True)  # (1, D_inner)

            tPredict = self.classifier(att_feat_sum)
            slide_sub_preds.append(tPredict)

            # patch-level CAM
            patch_cam = self._get_cam_1d(att_feats)  # (G, n_class)
            patch_softmax = torch.softmax(patch_cam, dim=1)
            _, sort_idx = torch.sort(patch_softmax[:, -1], descending=True)

            k = min(instance_per_group, len(sort_idx))
            topk_max = sort_idx[:k].long()
            topk_min = sort_idx[-k:].long()

            if self.distill == 'MaxMinS':
                topk_idx = torch.cat([topk_max, topk_min], dim=0)
                slide_pseudo_feat.append(mid_feat[topk_idx])
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(mid_feat[topk_max])
            else:  # AFS
                slide_pseudo_feat.append(att_feat_sum)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)

        # Second-tier: attention classifier on pseudo features
        logits, _, _ = self.attCls(slide_pseudo_feat)

        if label is not None and self.training:
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
            sub_labels = label.expand(slide_sub_preds.shape[0])
            inst_loss = self.inst_loss_fn(slide_sub_preds, sub_labels)
            return logits, inst_loss
        return logits
