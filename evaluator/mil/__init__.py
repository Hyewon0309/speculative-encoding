"""MIL (Multiple Instance Learning) components for slide-level classification.

The 9 aggregator implementations are vendored:

- ABMIL / CLAM / DSMIL / ILRA / Transformer / TransMIL — adapted from
  PathGen-1.6M/WSI_classification (https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology).
- DFTD / RRT / WiKG — adapted from mahmoodlab/MIL-Lab.

Vendoring keeps Tab. 1 reproducible without an external clone.
"""

from .utils import Struct, set_seed
from .engine import train_one_epoch, evaluate, compute_classification_metrics_torchmetrics
from .builder import build_net, make_conf

__all__ = [
    "Struct",
    "set_seed",
    "train_one_epoch",
    "evaluate",
    "compute_classification_metrics_torchmetrics",
    "build_net",
    "make_conf",
]
