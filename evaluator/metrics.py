"""Classification metrics shared by every speculative-encoding eval runner.

The runners print and aggregate just five scalars per fold/seed:

    acc, precision, recall, macro_f1, auroc

`weighted_f1` was previously emitted alongside `macro_f1` but the two were
easy to confuse in tables — only `macro_f1` is kept and used as the F1 column
of Tab. 1. Other downstream metrics (balanced accuracy, kappa, sklearn
classification_report) are not consumed and intentionally not produced.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all:   Union[List[int], np.ndarray],
    probs_all:   Optional[Union[List[float], np.ndarray]] = None,
    prefix:      str = "",
    roc_kwargs:  Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the six classification scalars used across the project.

    Args:
        targets_all : Ground truth labels of shape ``[N]``.
        preds_all   : Predicted labels of shape ``[N]``.
        probs_all   : Predicted probabilities for AUROC.
                      Binary: ``[N]`` (positive-class prob).
                      Multi-class: ``[N, C]``.
        prefix      : Optional prefix prepended to every metric key
                      (e.g. ``"lin_"`` for linear-probe metrics).
        roc_kwargs  : Extra kwargs forwarded to :func:`sklearn.metrics.roc_auc_score`.

    Returns:
        ``{prefix+acc, +precision, +recall, +macro_f1, +auroc}``.
        ``auroc`` is omitted when ``probs_all is None`` and set to NaN if
        sklearn refuses to compute it (single-class targets, etc.).
    """
    targets_all = np.asarray(targets_all)
    preds_all = np.asarray(preds_all)

    acc = accuracy_score(targets_all, preds_all)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_all, preds_all, average="macro", zero_division=0,
    )

    out: Dict[str, Any] = {
        f"{prefix}acc":       float(acc),
        f"{prefix}precision": float(p_macro),
        f"{prefix}recall":    float(r_macro),
        f"{prefix}macro_f1":  float(f1_macro),
    }

    if probs_all is not None:
        try:
            auroc = roc_auc_score(targets_all, np.asarray(probs_all), **(roc_kwargs or {}))
            out[f"{prefix}auroc"] = float(auroc)
        except Exception:
            out[f"{prefix}auroc"] = float("nan")

    return out


def get_eval_metrics_from_probs(
    probs:   Union[torch.Tensor, np.ndarray],
    labels:  Union[torch.Tensor, np.ndarray],
    n_class: int = 2,
    prefix:  str = "",
    # `get_report` was kept here for API stability; it is now ignored.
    get_report: bool = False,  # noqa: ARG001
) -> Dict[str, Any]:
    """Convenience wrapper: compute the six scalars from softmax outputs.

    Args:
        probs    : ``[N, C]`` softmax output (numpy or torch).
        labels   : ``[N]`` ground truth labels.
        n_class  : Number of classes (binary uses ``probs[:, 1]`` for AUROC,
                   multi-class uses one-vs-one macro AUROC).
        prefix   : Metric key prefix.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    probs = np.asarray(probs)
    labels = np.asarray(labels).ravel()

    preds = probs.argmax(axis=1)
    if n_class == 2:
        probs_for_auc = probs[:, 1]
        roc_kwargs: Dict[str, Any] = {}
    else:
        probs_for_auc = probs
        roc_kwargs = {"multi_class": "ovo", "average": "macro"}

    return get_eval_metrics(
        labels,
        preds,
        probs_all=probs_for_auc,
        prefix=prefix,
        roc_kwargs=roc_kwargs,
    )
