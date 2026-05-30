"""DS-MIL (from PathGen architecture.dsmil)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Classifier_1fc


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class BClassifier(nn.Module):
    def __init__(self, conf, dropout_v=0.0, nonlinear=True, passing_v=False):
        super().__init__()
        input_size = conf.D_feat
        output_class = conf.n_class
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, conf.D_inner),
                nn.ReLU(),
                nn.Linear(conf.D_inner, 128),
                nn.Tanh(),
            )
        else:
            self.q = nn.Linear(input_size, conf.D_inner)
        self.v = nn.Identity() if not passing_v else nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
        )
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):
        device = feats.device
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)
        A = torch.mm(Q, q_max.t()) / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))
        A = A.t()
        A_out = A
        A = F.softmax(A, dim=-1)
        B = torch.mm(A, V).unsqueeze(0)
        C = self.fcc(B).view(1, -1)
        return C, A_out, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super().__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x[0])
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A
