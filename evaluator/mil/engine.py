"""MIL train one epoch and evaluate (from PathGen engine)."""

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import get_eval_metrics_from_probs
from .utils import MetricLogger, SmoothedValue, adjust_learning_rate
from .builder import build_net

try:
    from timm.utils import accuracy
except ImportError:
    def accuracy(pred, target, topk=(1,)):
        return (pred.argmax(dim=1) == target).float().mean().unsqueeze(0),


def compute_classification_metrics_torchmetrics(
    probs: torch.Tensor,
    y_true: Union[torch.Tensor, np.ndarray],
    n_class: int = 2,
    prefix: str = "",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Wrapper around ``get_eval_metrics_from_probs`` that drops non-numeric keys.

    Returns: ``{prefix+acc, +precision, +recall, +macro_f1, +auroc}``.
    """
    metrics = get_eval_metrics_from_probs(probs, y_true, n_class=n_class, prefix=prefix)
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def _loss_forward_backward(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger):
    preds = net(image_patches)
    ce_loss = criterion(preds, labels)
    loss = getattr(conf, "w_loss", 1.0) * 0.0 + ce_loss
    loss.backward()
    metric_logger.update(lr=optimizer.param_groups[0]["lr"], ce_loss=ce_loss.item())


def _loss_forward_backward_dftd(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger):
    logits, inst_loss = net(image_patches, label=labels)
    ce_loss = criterion(logits, labels)
    bag_weight = getattr(conf, "bag_weight", 0.7)
    loss = bag_weight * ce_loss + (1 - bag_weight) * inst_loss
    loss.backward()
    metric_logger.update(lr=optimizer.param_groups[0]["lr"], ce_loss=ce_loss.item(), inst_loss=inst_loss.item())


def _loss_forward_backward_dsmil(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger):
    ins_preds, bag_preds, attn = net(image_patches)
    max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
    ce_loss = 0.5 * criterion(max_preds, labels) + 0.5 * criterion(bag_preds, labels)
    diff_loss = torch.tensor(0.0, device=device)
    attn = torch.softmax(attn, dim=-1)
    n_token = getattr(conf, "n_token", 1)
    for i in range(n_token):
        for j in range(i + 1, n_token):
            diff_loss = diff_loss + torch.cosine_similarity(attn[i], attn[j], dim=-1).mean() / max(1, n_token * (n_token - 1) / 2)
    loss = getattr(conf, "w_loss", 1.0) * diff_loss + ce_loss
    loss.backward()
    metric_logger.update(lr=optimizer.param_groups[0]["lr"], ce_loss=ce_loss.item(), diff_loss=diff_loss.item())


def _loss_forward_backward_clam(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger):
    logits, instance_loss = net(image_patches, labels, instance_eval=True)
    loss = criterion(logits, labels)
    total_loss = getattr(conf, "w_loss", 1.0) * loss + (1 - getattr(conf, "w_loss", 1.0)) * instance_loss
    total_loss.backward()
    metric_logger.update(lr=optimizer.param_groups[0]["lr"], bag_loss=loss.item(), instance_loss=instance_loss.item())


def train_one_epoch(net, criterion, data_loader, optimizer, device, epoch, conf, log_writer=None, verbose=False):
    import time
    net.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    iterable = (
        metric_logger.log_every(data_loader, 100, f"Epoch: [{epoch}]")
        if verbose
        else data_loader
    )
    t0 = time.time()
    for data_it, data in enumerate(iterable):
        image_patches = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device)
        coords = data.get("coords")

        adjust_learning_rate(optimizer, epoch + data_it / max(1, len(data_loader)), conf)
        optimizer.zero_grad()

        if conf.arch == "dsmil":
            _loss_forward_backward_dsmil(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger)
        elif conf.arch in ("clam_sb", "clam_mb"):
            _loss_forward_backward_clam(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger)
        elif conf.arch == "dftd":
            _loss_forward_backward_dftd(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger)
        elif conf.arch == "bmil_spvis":
            raise NotImplementedError("bmil_spvis not supported in vendored mil")
        else:
            _loss_forward_backward(net, image_patches, labels, criterion, conf, device, optimizer, metric_logger)

        optimizer.step()

    if not verbose:
        for key in ("ce_loss", "loss", "bag_loss"):
            if key in metric_logger.meters:
                loss_val = metric_logger.meters[key].global_avg
                break
        else:
            loss_val = 0.0
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch:3d}]  train  loss={loss_val:.4f}  lr={lr:.6f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate(
    net,
    criterion,
    data_loader,
    device,
    conf,
    header,
    verbose=False,
    return_details=False,
    measure_gpu_time=False,
):
    net.eval()
    y_pred, y_true = [], []
    metric_logger = MetricLogger(delimiter="  ")
    iterable = metric_logger.log_every(data_loader, 100, header) if verbose else data_loader
    gpu_time_ms = 0.0
    use_gpu_timer = bool(measure_gpu_time and str(device).startswith("cuda") and torch.cuda.is_available())
    selection_time_seconds = 0.0

    for data in iterable:
        image_patches = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device)
        coords = data.get("coords")
        if "selection_time" in data:
            selection_time_seconds += float(data["selection_time"].reshape(-1)[0].item())

        if use_gpu_timer:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if conf.arch == "dsmil":
            ins_preds, bag_preds, attn = net(image_patches)
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            loss = 0.5 * criterion(max_preds, labels) + 0.5 * criterion(bag_preds, labels)
            pred = 0.5 * torch.softmax(max_preds, dim=-1) + 0.5 * torch.softmax(bag_preds, dim=-1)
        elif conf.arch == "bmil_spvis":
            raise NotImplementedError("bmil_spvis not supported in vendored mil")
        elif conf.arch in ("clam_sb", "clam_mb"):
            output = net(image_patches)
            loss = criterion(output, labels)
            pred = torch.softmax(output, dim=-1)
        else:
            output = net(image_patches)
            loss = criterion(output, labels)
            pred = torch.softmax(output, dim=-1)
        if use_gpu_timer:
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms += float(start_event.elapsed_time(end_event))

        acc1 = accuracy(pred, labels, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=labels.shape[0])
        y_pred.append(pred)
        y_true.append(labels)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    n_class = getattr(conf, "n_class", 2)
    m = compute_classification_metrics_torchmetrics(y_pred, y_true, n_class=n_class, device=device)
    auroc = m["auroc"]
    acc = m["acc"]
    f1_score = m["macro_f1"]
    precision, recall = m["precision"], m["recall"]
    # acc1: timm returns 0-100, fallback returns 0-1
    acc_pct = metric_logger.acc1.global_avg if metric_logger.acc1.global_avg > 1.0 else metric_logger.acc1.global_avg * 100.0
    loss_avg = metric_logger.loss.global_avg
    if not verbose:
        print(f"  {header:4}  acc={acc_pct:5.1f}%  loss={loss_avg:.3f}  auroc={auroc:.3f}  prec={precision:.3f}  rec={recall:.3f}  f1={f1_score:.3f}")
    else:
        print(f"* Acc@1 {metric_logger.acc1.global_avg:.3f} loss {loss_avg:.3f} auroc {auroc:.3f} prec {precision:.3f} rec {recall:.3f} f1 {f1_score:.3f}")
    if return_details:
        out = (
            auroc,
            metric_logger.acc1.global_avg,
            f1_score,
            loss_avg,
            precision,
            recall,
            m,
            y_pred.detach().cpu(),
            y_true.detach().cpu(),
        )
        if use_gpu_timer:
            return out + (gpu_time_ms / 1000.0, selection_time_seconds)
        return out
    return auroc, metric_logger.acc1.global_avg, f1_score, loss_avg, precision, recall
