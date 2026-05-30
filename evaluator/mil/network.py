"""Linear/MLP blocks for MIL (from PathGen architecture.network)."""

import torch
import torch.nn as nn


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if droprate != 0.0:
            self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        return self.fc(x)


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res
        self.resBlocks = nn.Sequential(*[residual_block(m_dim) for _ in range(numLayer_Res)])

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        if self.numRes > 0:
            x = self.resBlocks(x)
        return x


class DimReduction1(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res
        self.resBlocks = nn.Sequential(*[residual_block(m_dim) for _ in range(numLayer_Res)])

    def forward(self, x):
        x_ = x
        x = self.relu1(self.fc1(x) + x_)
        if self.numRes > 0:
            x = self.resBlocks(x)
        return x
