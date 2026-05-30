"""Shared utilities: seeding, logging, autocast, warnings, LR schedules."""

import logging
import math
import os
import random
import time
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch

from distill_lib.distributed import is_main_process


def set_seed(seed: int, rank: int):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def get_logger(name: str = "distill", log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    effective_level = logging.INFO if is_main_process() else logging.WARNING
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(effective_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_path and is_main_process():
        log_path = os.path.abspath(log_path)
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == log_path
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


class WandbTracker:
    """Thin main-process-only wrapper around wandb."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.run = None

    def init(self, args, output_dir: str):
        if not self.enabled or not is_main_process():
            return
        import wandb

        tags = list(args.wandb_tags) if args.wandb_tags else None
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags,
            dir=output_dir,
            config=vars(args),
        )

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if not self.run or not metrics:
            return
        payload = {k: v for k, v in metrics.items() if v is not None}
        if payload:
            self.run.log(payload, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


@contextmanager
def timed_stage(logger: logging.Logger | None, name: str):
    start = time.perf_counter()
    if logger and is_main_process():
        logger.info("[stage/start] %s", name)
    try:
        yield
    finally:
        if logger and is_main_process():
            elapsed = time.perf_counter() - start
            logger.info("[stage/done] %s | %.2fs", name, elapsed)


def get_autocast(device, precision: str = "bf16"):
    """Return an autocast context manager for the given precision.

    Args:
        device: torch device.
        precision: "bf16", "fp16", or "fp32" (no autocast).
    """
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def get_grad_scaler(device, precision: str = "bf16"):
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _configure_warnings(args):
    if not args.suppress_warnings:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Support for mismatched src_key_padding_mask and mask is deprecated.*",
        category=UserWarning,
    )
    master_addr = os.environ.get("MASTER_ADDR", "")
    if master_addr.lower() in ("", "localhost"):
        os.environ["MASTER_ADDR"] = "127.0.0.1"


def _build_cosine_schedule(step: int, max_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if max_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(warmup_steps, 1))
    if warmup_steps >= max_steps:
        return 1.0
    progress = float(step - warmup_steps) / float(max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return max(min_lr_ratio, cosine)
