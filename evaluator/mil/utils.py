"""MIL training utilities (vendored from PathGen utils.utils)."""

import math
import random
import time
from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < getattr(cfg, "warmup_epoch", 0):
        lr = cfg.lr * epoch / max(1, cfg.warmup_epoch)
    else:
        n = max(1, getattr(cfg, "train_epoch", 50) - getattr(cfg, "warmup_epoch", 0))
        lr = getattr(cfg, "min_lr", 0) + (cfg.lr - getattr(cfg, "min_lr", 0)) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - getattr(cfg, "warmup_epoch", 0)) / n)
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr


def softmax_one(x, dim=-1):
    exp_x = torch.exp(x)
    denominator = exp_x.sum(dim=dim, keepdim=True) + 1
    return exp_x / denominator


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    def __str__(self):
        median = np.median(list(self.deque)) if self.deque else 0.0
        return self.fmt.format(
            median=median,
            global_avg=self.global_avg,
            avg=self.global_avg,
            value=median,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(lambda: SmoothedValue())
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())

    def log_every(self, iterable, print_freq, header=None):
        if header is None:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        i = 0
        n = len(iterable)
        for obj in iterable:
            yield obj
            i += 1
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n:
                eta = iter_time.global_avg * (n - i)
                eta_str = str(timedelta(seconds=int(eta)))
                if torch.cuda.is_available():
                    print(
                        f"{header}[{i}/{n}] eta: {eta_str}  {self}  time: {iter_time}  "
                        f"max mem: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f}"
                    )
                else:
                    print(f"{header}[{i}/{n}] eta: {eta_str}  {self}  time: {iter_time}")
            end = time.time()
        total = time.time() - start_time
        print(f"{header} Total time: {timedelta(seconds=int(total))} ({total / max(1, n):.4f} s/it)")


def save_model(conf, model, optimizer, epoch, save_path=None, **kwargs):
    """Save checkpoint. save_path overrides conf.ckpt_dir if provided."""
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": getattr(conf, "__dict__", conf),
    }
    to_save.update(kwargs)
    path = save_path
    if path is None:
        ckpt_dir = getattr(conf, "ckpt_dir", ".")
        path = f"{ckpt_dir}/checkpoint-{epoch}.pth"
    path = str(path)
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(to_save, path)
