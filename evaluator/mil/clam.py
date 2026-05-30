"""CLAM-SB / CLAM-MB (from PathGen architecture.clam)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import initialize_weights, softmax_one


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        mod = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            mod.append(nn.Dropout(0.25))
        mod.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*mod)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            *(nn.Dropout(0.25),) if dropout else (),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            *(nn.Dropout(0.25),) if dropout else (),
        )
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        A = self.attention_c(self.attention_a(x) * self.attention_b(x))
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, conf, gate=True, size_arg="small", k_sample=8, dropout=True,
                 instance_loss_fn=None):
        super().__init__()
        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        n_classes = conf.n_class
        self.size_dict = {"small": [conf.D_feat, conf.D_inner, 128], "big": [conf.D_feat, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.append(Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1) if gate else Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1))
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = conf.n_class > 2
        initialize_weights(self)

    def relocate(self):
        pass  # caller moves to device

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        if A.dim() == 1:
            A = A.unsqueeze(0)
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        all_targets = torch.cat([self.create_positive_targets(k, h.device),
                                 self.create_negative_targets(k, h.device)], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, torch.topk(logits, 1, dim=1)[1].squeeze(1), all_targets

    def inst_eval_out(self, A, h, classifier):
        if A.dim() == 1:
            A = A.unsqueeze(0)
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k, h.device)
        logits = classifier(top_p)
        return self.instance_loss_fn(logits, p_targets), torch.topk(logits, 1, dim=1)[1].squeeze(1), p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h[0])
        A = A.transpose(-1, -2)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=-1)
        if instance_eval and label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            if inst_labels.dim() == 0:
                inst_labels = inst_labels.unsqueeze(0)
            for i in range(len(self.instance_classifiers)):
                # CLAM-SB: single attention head → always use A[0]
                a_i = A[min(i, A.shape[0] - 1)]
                if inst_labels[i].item() == 1:
                    total_inst_loss += self.inst_eval(a_i, h, self.instance_classifiers[i])[0]
                elif self.subtyping:
                    total_inst_loss += self.inst_eval_out(a_i, h, self.instance_classifiers[i])[0]
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        if instance_eval and label is not None:
            return logits, total_inst_loss
        return logits


class CLAM_MB(CLAM_SB):
    def __init__(self, conf, gate=True, size_arg="small", k_sample=8, dropout=True,
                 instance_loss_fn=None):
        nn.Module.__init__(self)
        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        n_classes = conf.n_class
        self.size_dict = {"small": [conf.D_feat, conf.D_inner, 128], "big": [conf.D_feat, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.append(Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes) if gate else Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes))
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = n_classes > 2
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h[0])
        A = A.transpose(-1, -2)
        if attention_only:
            return A
        A_raw = A
        A = softmax_one(A, dim=1)
        if instance_eval and label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                if inst_labels[i].item() == 1:
                    total_inst_loss += self.inst_eval(A[i], h, self.instance_classifiers[i])[0]
                elif self.subtyping:
                    total_inst_loss += self.inst_eval_out(A[i], h, self.instance_classifiers[i])[0]
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes, dtype=torch.float32, device=h.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        if instance_eval and label is not None:
            return logits, total_inst_loss
        return logits
