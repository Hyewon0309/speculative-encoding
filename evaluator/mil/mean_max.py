"""Mean/Max MIL (from PathGen modules.mean_max)."""

import torch.nn as nn

from .utils import initialize_weights


class MeanMIL(nn.Module):
    def __init__(self, conf, dropout=True, act="relu", test=False):
        super().__init__()
        head = [nn.Linear(conf.D_feat, conf.D_inner)]
        if act.lower() == "relu":
            head += [nn.ReLU()]
        elif act.lower() == "gelu":
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(conf.D_inner, conf.n_class)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.head(x).mean(dim=1)


class MaxMIL(nn.Module):
    def __init__(self, conf, dropout=True, act="relu", test=False):
        super().__init__()
        head = [nn.Linear(conf.D_feat, conf.D_inner)]
        if act.lower() == "relu":
            head += [nn.ReLU()]
        elif act.lower() == "gelu":
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(conf.D_inner, conf.n_class)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.head(x).max(dim=1)[0]
