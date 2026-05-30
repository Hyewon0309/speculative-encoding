"""Evaluation utilities used across the speculative-encoding runners.

The package re-exports only the metrics helpers that the eval CLI and the
per-model runners rely on. Per-arch MIL training/evaluation code lives under
``evaluator/mil/`` and is loaded through ``evaluator.mil.build_net`` /
``evaluator.mil.evaluate``.
"""

from .metrics import get_eval_metrics, get_eval_metrics_from_probs

__all__ = [
    "get_eval_metrics",
    "get_eval_metrics_from_probs",
]
