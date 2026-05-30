"""Linear-probe metrics for the distillation evaluator.

Mirrors ``evaluator/metrics.py`` (one extra: ``auc`` is multi-class one-vs-rest
since the linear probe runs on the CRC-100K classification benchmark). Only
macro-averaged precision/recall/F1 are emitted — the weighted variants and
balanced accuracy were unused downstream and have been removed to match the
main eval runners.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def _to_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().float().cpu().numpy()
    return np.asarray(array)


def compute_classification_metrics(logits, labels, label_names=None):
    """Compute the linear-probe scalars logged at every distillation eval step."""
    probs = _softmax(_to_numpy(logits))
    y_true = _to_numpy(labels)
    preds = probs.argmax(axis=1)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0,
    )
    metrics = {
        "accuracy":        float(accuracy_score(y_true, preds)),
        "precision_macro": float(p),
        "recall_macro":    float(r),
        "f1_macro":        float(f1),
    }

    try:
        metrics["auc"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", labels=list(range(probs.shape[1])))
        )
    except ValueError:
        metrics["auc"] = None

    if label_names:
        metrics["label_names"] = label_names
    return metrics
