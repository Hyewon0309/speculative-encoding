"""Distiller: main distillation training loop."""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from distill_lib.distributed import barrier, get_rank, get_world_size, is_main_process
from distill_lib.encoder import _drop_prefix_tokens, _extract_sequence, _unwrap_ddp
from distill_lib.optim import build_optimizers
from distill_lib.utils import _build_cosine_schedule, get_autocast, get_grad_scaler


class Distiller:
    def __init__(self, args, teacher, student, projector, device, logger,
                 linear_probe, teacher_prefix_tokens, student_prefix_tokens,
                 student_config, tracker=None, pixel_normalize=None,
                 cka_loader=None):
        self.args = args
        self.teacher = teacher
        self.student = student
        self.projector = projector
        self.device = device
        self.logger = logger
        self.linear_probe = linear_probe
        self.tracker = tracker
        self.teacher_prefix_tokens = teacher_prefix_tokens
        self.student_prefix_tokens = student_prefix_tokens
        self.student_config = student_config
        self.pixel_normalize = pixel_normalize
        self.cka_loader = cka_loader

        self.optimizers, self.clip_params = build_optimizers(
            args,
            student,
            projector,
        )
        self.has_projector = projector is not None and any(p.requires_grad for p in projector.parameters())
        # Use the requested mixed-precision mode for both frozen teacher and
        # trainable student. Structural losses still upcast selected math-heavy
        # comparisons to fp32 later in the loss path where needed.
        self.teacher_autocast = get_autocast(device, precision=args.precision)
        self.student_autocast = get_autocast(device, precision=args.precision)
        self.scaler = get_grad_scaler(device, precision=args.precision)
        self.eps = 1e-6
        if self.logger and is_main_process():
            optimizer_names = ", ".join(name for name, _ in self.optimizers)
            self.logger.info("Optimizers: %s", optimizer_names)

    def _structure_regression(self, student_value, teacher_value):
        if self.args.structure_loss_type == "smooth_l1":
            return F.smooth_l1_loss(student_value, teacher_value)
        return F.mse_loss(student_value, teacher_value)

    def _distribution_loss(self, student_logits, teacher_logits, temperature, exclude_diagonal=False):
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        temperature = max(temperature, self.eps)
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature

        if exclude_diagonal and student_scaled.dim() >= 2 and student_scaled.size(-1) == student_scaled.size(-2):
            diag = torch.eye(student_scaled.size(-1), device=student_scaled.device, dtype=torch.bool)
            while diag.dim() < student_scaled.dim():
                diag = diag.unsqueeze(0)
            fill_value = torch.finfo(student_scaled.dtype).min
            student_scaled = student_scaled.masked_fill(diag, fill_value)
            teacher_scaled = teacher_scaled.masked_fill(diag, fill_value)

        teacher_prob = F.softmax(teacher_scaled, dim=-1)
        student_log_prob = F.log_softmax(student_scaled, dim=-1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="none")

        return loss.sum(dim=-1).mean() * (temperature ** 2)

    def _relation_loss(self, student_value, teacher_value, temperature, is_distance=False, exclude_diagonal=False):
        if self.args.structure_loss_type == "kl":
            if is_distance:
                return self._distribution_loss(-student_value, -teacher_value, temperature, exclude_diagonal=exclude_diagonal)
            return self._distribution_loss(student_value, teacher_value, temperature, exclude_diagonal=exclude_diagonal)
        return self._structure_regression(student_value, teacher_value)

    def _all_gather_with_local_grad(self, tensor):
        if not self.args.distributed:
            return tensor.contiguous()
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
        if not tensor.requires_grad:
            dist.all_gather(gathered, tensor)
        else:
            dist.all_gather(gathered, tensor.detach())
            gathered[get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    def _scale_vectors(self, x, reduce_dim):
        norms = x.norm(dim=-1, keepdim=True)
        scale = norms.mean(dim=reduce_dim, keepdim=True).clamp_min(self.eps)
        return x / scale, scale

    def _reduce_metrics_for_logging(self, metrics):
        reduced = {}
        if not metrics:
            return reduced
        numeric_keys = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        if not numeric_keys:
            return reduced
        values = torch.tensor(
            [float(metrics[k]) for k in numeric_keys],
            device=self.device,
            dtype=torch.float32,
        )
        if self.args.distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values = values / get_world_size()
        reduced.update({k: float(v) for k, v in zip(numeric_keys, values.tolist())})
        for k, v in metrics.items():
            if k not in reduced and v is not None:
                reduced[k] = v
        return reduced

    def _affinity_matrix(self, x, temperature):
        x_scaled, scale = self._scale_vectors(x.float(), reduce_dim=-2 if x.dim() == 3 else 0)
        affinity = x_scaled @ x_scaled.transpose(-1, -2)
        return affinity / max(temperature, self.eps), scale

    def _distance_matrix(self, x, temperature):
        x_scaled, scale = self._scale_vectors(x.float(), reduce_dim=-2 if x.dim() == 3 else 0)
        distance = torch.cdist(x_scaled, x_scaled)
        return distance / max(temperature, self.eps), scale

    def _raw_affinity_and_distance(self, x):
        """Compute scale-normalized gram and distance matrices without temperature.

        Returns ``(gram, raw_distance, scale)`` so callers can apply different
        temperatures cheaply without recomputing the O(N²d) gram product.
        """
        x_scaled, scale = self._scale_vectors(x.float(), reduce_dim=-2 if x.dim() == 3 else 0)
        gram = x_scaled @ x_scaled.transpose(-1, -2)
        norms_sq = gram.diagonal(dim1=-2, dim2=-1)
        dist_sq = norms_sq.unsqueeze(-1) + norms_sq.unsqueeze(-2) - 2 * gram
        dist_sq = dist_sq.clamp_min(0)
        if dist_sq.size(-1) == dist_sq.size(-2):
            diag = torch.eye(dist_sq.size(-1), device=dist_sq.device, dtype=torch.bool)
            while diag.dim() < dist_sq.dim():
                diag = diag.unsqueeze(0)
            dist_sq = dist_sq.masked_fill(diag, 1.0)
        # Guard sqrt backward: grad of sqrt(x) at x→0 is 1/(2√x) → inf.
        # Off-diagonal entries can be near-zero for similar CLS tokens (especially
        # at initialisation), producing inf gradients that poison the GradScaler.
        raw_distance = (dist_sq + self.eps).sqrt()
        return gram, raw_distance, scale

    def _affinity_and_distance(self, x, temperature):
        """Compute both affinity and distance from a single scaling + gram matrix."""
        gram, raw_distance, scale = self._raw_affinity_and_distance(x)
        temp = max(temperature, self.eps)
        return gram / temp, raw_distance / temp, scale

    def _safe_pairwise_distance(self, x):
        """Pairwise L2 distance with a safe sqrt backward and zero diagonal.

        ``torch.cdist(X, X)`` has zero on the diagonal, and sqrt's gradient is
        infinite at 0 — that poisons the backward pass (and silently stalls
        fp16 training via GradScaler step skipping). Mirror the guard used in
        ``_raw_affinity_and_distance``: clamp, replace the diagonal with 1
        before sqrt, add eps, then re-zero the diagonal afterwards.
        """
        x = x.float()
        gram = x @ x.transpose(-1, -2)
        norms_sq = gram.diagonal(dim1=-2, dim2=-1)
        dist_sq = norms_sq.unsqueeze(-1) + norms_sq.unsqueeze(-2) - 2 * gram
        dist_sq = dist_sq.clamp_min(0)
        diag = torch.eye(dist_sq.size(-1), device=dist_sq.device, dtype=torch.bool)
        while diag.dim() < dist_sq.dim():
            diag = diag.unsqueeze(0)
        dist_sq = dist_sq.masked_fill(diag, 1.0)
        distance = (dist_sq + self.eps).sqrt()
        return distance.masked_fill(diag, 0.0)

    def _cls_anchor_relations(self, cls_token, patch_tokens):
        cls_scaled, cls_scale = self._scale_vectors(cls_token.float(), reduce_dim=0)
        patch_scaled, patch_scale = self._scale_vectors(patch_tokens.float(), reduce_dim=-2)
        affinity = (
            patch_scaled * cls_scaled.unsqueeze(1)
        ).sum(dim=-1)
        delta = patch_tokens.float() - cls_token.unsqueeze(1).float()
        radial = delta.norm(dim=-1) / (patch_scale.squeeze(-1) + self.eps)
        cls_norm = cls_token.float().norm(dim=-1) / (cls_scale.squeeze(-1) + self.eps)
        return affinity, radial, cls_norm

    def _forward_teacher(self, pixel_values):
        """Frozen forward -> (cls (B, 1536), patches (B, 256, 1536))."""
        with torch.inference_mode():
            with self.teacher_autocast:
                outputs = self.teacher(pixel_values)
                tokens = _extract_sequence(outputs)
                cls_token = tokens[:, 0]
                patches = _drop_prefix_tokens(tokens, self.teacher_prefix_tokens)
        return cls_token.clone(), patches.clone()

    def _forward_student(self, pixel_values):
        """Trainable forward -> (cls (B, student_dim), patches (B, 256, student_dim))."""
        with self.student_autocast:
            outputs = self.student(pixel_values)
            tokens = _extract_sequence(outputs)
            cls_token = tokens[:, 0]
            patches = _drop_prefix_tokens(tokens, self.student_prefix_tokens)
        return cls_token, patches

    def _compute_loss(
        self,
        student_cls,
        student_tokens,
        teacher_cls,
        teacher_tokens,
    ):
        with self.student_autocast:
            backward_loss = torch.tensor(0.0, device=self.device)
            reported_loss = torch.tensor(0.0, device=self.device)
            metrics = {}

            projector = None
            if self.has_projector:
                projector = _unwrap_ddp(self.projector) if isinstance(self.projector, DDP) else self.projector

            if self.args.feat_loss_weight > 0 and projector is not None:
                projected_tokens = projector.project_tokens(student_tokens)
                loss_feat = F.mse_loss(projected_tokens, teacher_tokens)
                weighted_loss = self.args.feat_loss_weight * loss_feat
                backward_loss = backward_loss + weighted_loss
                reported_loss = reported_loss + weighted_loss
                metrics["train/loss_feat"] = float(loss_feat.detach())

            if self.args.cls_loss_weight > 0 and projector is not None:
                projected_cls = projector.project_pool(student_cls)
                loss_cls = F.mse_loss(projected_cls, teacher_cls)
                weighted_loss = self.args.cls_loss_weight * loss_cls
                backward_loss = backward_loss + weighted_loss
                reported_loss = reported_loss + weighted_loss
                metrics["train/loss_cls"] = float(loss_cls.detach())

            if self.args.cosine_loss_weight > 0 and projector is not None:
                student_pool = student_tokens.mean(dim=1)
                teacher_pool = teacher_tokens.mean(dim=1)
                projected_pool = projector.project_pool(student_pool)
                loss_cos = 1.0 - F.cosine_similarity(projected_pool, teacher_pool, dim=-1).mean()
                weighted_loss = self.args.cosine_loss_weight * loss_cos
                backward_loss = backward_loss + weighted_loss
                reported_loss = reported_loss + weighted_loss
                metrics["train/loss_cosine"] = float(loss_cos.detach())

            # --- CLS-anchored patch relations: dot-product + radial magnitude ---
            if self.args.cls_structure_loss_weight > 0:
                student_anchor_affinity, student_anchor_radius, student_cls_norm = (
                    self._cls_anchor_relations(student_cls, student_tokens)
                )
                teacher_anchor_affinity, teacher_anchor_radius, teacher_cls_norm = (
                    self._cls_anchor_relations(teacher_cls, teacher_tokens)
                )
                temperatures = self.args.structure_temperature
                n_temps = len(temperatures)
                loss_cls_affinity = torch.tensor(0.0, device=self.device)
                loss_cls_radius = torch.tensor(0.0, device=self.device)
                use_affinity = not self.args.only_distance
                use_distance = not self.args.only_affinity
                for temp in temperatures:
                    if use_affinity:
                        loss_cls_affinity = loss_cls_affinity + self._relation_loss(
                            student_anchor_affinity, teacher_anchor_affinity, temp,
                        )
                    if use_distance:
                        loss_cls_radius = loss_cls_radius + self._relation_loss(
                            student_anchor_radius, teacher_anchor_radius, temp, is_distance=True,
                        )
                loss_cls_affinity = loss_cls_affinity / n_temps
                loss_cls_radius = loss_cls_radius / n_temps
                if use_distance:
                    loss_cls_norm = self._structure_regression(student_cls_norm, teacher_cls_norm)
                else:
                    loss_cls_norm = torch.tensor(0.0, device=self.device)
                loss_cls_structure = loss_cls_affinity + loss_cls_radius + 0.25 * loss_cls_norm
                weighted_loss = self.args.cls_structure_loss_weight * loss_cls_structure
                backward_loss = backward_loss + weighted_loss
                reported_loss = reported_loss + weighted_loss
                metrics["train/loss_cls_structure"] = float(loss_cls_structure.detach())
                metrics["train/loss_cls_anchor_affinity"] = float(loss_cls_affinity.detach())
                metrics["train/loss_cls_anchor_radius"] = float(loss_cls_radius.detach())

            # --- Intra-image patch topology relative to CLS: affinity + distance ---
            if self.args.structure_loss_weight > 0:
                student_centered = student_tokens.float() - student_cls.float().unsqueeze(1)
                teacher_centered = teacher_tokens.float() - teacher_cls.float().unsqueeze(1)
                s_gram, s_raw_dist, _ = self._raw_affinity_and_distance(student_centered)
                t_gram, t_raw_dist, _ = self._raw_affinity_and_distance(teacher_centered)
                temperatures = self.args.structure_temperature
                n_temps = len(temperatures)
                loss_structure_affinity = torch.tensor(0.0, device=self.device)
                loss_structure_distance = torch.tensor(0.0, device=self.device)
                use_affinity = not self.args.only_distance
                use_distance = not self.args.only_affinity
                for temp in temperatures:
                    if use_affinity:
                        loss_structure_affinity = loss_structure_affinity + self._relation_loss(
                            s_gram, t_gram, temp, exclude_diagonal=True,
                        )
                    if use_distance:
                        loss_structure_distance = loss_structure_distance + self._relation_loss(
                            s_raw_dist, t_raw_dist, temp, is_distance=True, exclude_diagonal=True,
                        )
                loss_structure_affinity = loss_structure_affinity / n_temps
                loss_structure_distance = loss_structure_distance / n_temps
                loss_structure = loss_structure_affinity + loss_structure_distance
                weighted_loss = self.args.structure_loss_weight * loss_structure
                backward_loss = backward_loss + weighted_loss
                reported_loss = reported_loss + weighted_loss
                metrics["train/loss_structure"] = float(loss_structure.detach())
                metrics["train/loss_structure_affinity"] = float(loss_structure_affinity.detach())
                metrics["train/loss_structure_distance"] = float(loss_structure_distance.detach())
                metrics["train/patch_affinity_alignment"] = float(
                    F.cosine_similarity(
                        s_gram.flatten(1), t_gram.flatten(1), dim=-1,
                    ).mean().detach()
                )

            # --- Global batch CLS relations across all GPUs ---
            if self.args.new_loss or self.args.global_structure_loss_weight > 0:
                global_student_cls = self._all_gather_with_local_grad(student_cls.float())
                global_teacher_cls = self._all_gather_with_local_grad(teacher_cls.float())
                if self.args.new_loss:
                    student_dist = self._safe_pairwise_distance(global_student_cls)
                    teacher_dist = self._safe_pairwise_distance(global_teacher_cls)
                    loss_pairwise = F.mse_loss(student_dist, teacher_dist)
                    weighted_loss = self.args.new_loss_weight * loss_pairwise
                    reported_loss = reported_loss + weighted_loss
                    # The gathered tensor keeps autograd only for the local rank slice.
                    # DDP averages parameter gradients across ranks afterward, so recover
                    # the intended global-batch gradient magnitude on the backward path.
                    if self.args.distributed:
                        weighted_loss = weighted_loss * get_world_size()
                    backward_loss = backward_loss + weighted_loss
                    metrics["train/loss_pairwise_l2"] = float(loss_pairwise.detach())
                else:
                    s_gram, s_raw_dist, _ = self._raw_affinity_and_distance(global_student_cls)
                    t_gram, t_raw_dist, _ = self._raw_affinity_and_distance(global_teacher_cls)
                    temperatures = self.args.structure_temperature
                    n_temps = len(temperatures)
                    loss_global_affinity = torch.tensor(0.0, device=self.device)
                    loss_global_distance = torch.tensor(0.0, device=self.device)
                    use_affinity = not self.args.only_distance
                    use_distance = not self.args.only_affinity
                    for temp in temperatures:
                        if use_affinity:
                            loss_global_affinity = loss_global_affinity + self._relation_loss(
                                s_gram, t_gram, temp, exclude_diagonal=True,
                            )
                        if use_distance:
                            loss_global_distance = loss_global_distance + self._relation_loss(
                                s_raw_dist, t_raw_dist, temp, is_distance=True, exclude_diagonal=True,
                            )
                    loss_global_affinity = loss_global_affinity / n_temps
                    loss_global_distance = loss_global_distance / n_temps
                    loss_global_structure = loss_global_affinity + loss_global_distance
                    weighted_loss = self.args.global_structure_loss_weight * loss_global_structure
                    reported_loss = reported_loss + weighted_loss
                    # The gathered tensor keeps autograd only for the local rank slice.
                    # DDP averages parameter gradients across ranks afterward, so recover
                    # the intended global-batch gradient magnitude on the backward path.
                    if self.args.distributed:
                        weighted_loss = weighted_loss * get_world_size()
                    backward_loss = backward_loss + weighted_loss
                    metrics["train/loss_global_structure"] = float(loss_global_structure.detach())
                    metrics["train/loss_global_affinity"] = float(loss_global_affinity.detach())
                    metrics["train/loss_global_distance"] = float(loss_global_distance.detach())
                    metrics["train/global_cls_alignment"] = float(
                        F.cosine_similarity(
                            s_gram.flatten().unsqueeze(0),
                            t_gram.flatten().unsqueeze(0),
                            dim=-1,
                        ).mean().detach()
                    )

        metrics["train/loss_total"] = float(reported_loss.detach())
        return backward_loss, metrics

    def _measure_cka(self, step):
        """Linear CKA between teacher and student CLS features over a fixed
        train-image subset. All ranks forward their shard in parallel; sums of
        squares are all-reduced before the final closed-form CKA is computed on
        rank 0. Memory per rank is O(d²) rather than O(N·d).
        """
        if self.cka_loader is None:
            return

        teacher_module = _unwrap_ddp(self.teacher)
        student_module = _unwrap_ddp(self.student)
        was_training = getattr(student_module, "training", False)
        if hasattr(teacher_module, "eval"):
            teacher_module.eval()
        if hasattr(student_module, "eval"):
            student_module.eval()

        sum_t = sum_s = None
        sum_tt = sum_ss = sum_ts = None
        n_total = 0

        with torch.no_grad():
            for batch in self.cka_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                if self.pixel_normalize is not None:
                    pixel_values = self.pixel_normalize(pixel_values)
                with self.teacher_autocast:
                    t_tokens = _extract_sequence(teacher_module(pixel_values))
                with self.student_autocast:
                    s_tokens = _extract_sequence(student_module(pixel_values))
                t_feat = t_tokens[:, 0].float()
                s_feat = s_tokens[:, 0].float()

                b_sum_t = t_feat.sum(dim=0)
                b_sum_s = s_feat.sum(dim=0)
                b_tt = t_feat.T @ t_feat
                b_ss = s_feat.T @ s_feat
                b_ts = t_feat.T @ s_feat
                if sum_t is None:
                    sum_t, sum_s = b_sum_t, b_sum_s
                    sum_tt, sum_ss, sum_ts = b_tt, b_ss, b_ts
                else:
                    sum_t = sum_t + b_sum_t
                    sum_s = sum_s + b_sum_s
                    sum_tt = sum_tt + b_tt
                    sum_ss = sum_ss + b_ss
                    sum_ts = sum_ts + b_ts
                n_total += t_feat.size(0)

        if was_training and hasattr(student_module, "train"):
            student_module.train()

        # ── All-reduce across ranks (DistributedSampler shards the subset) ──
        # Every rank must participate in the collective, so if a rank happened
        # to get zero batches we skip the reduce and bail on all ranks.
        if self.args.distributed:
            has_batch = torch.tensor(
                [1.0 if n_total > 0 else 0.0], device=self.device,
            )
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
            if has_batch.item() < 1.0:
                if self.logger and is_main_process():
                    self.logger.warning(
                        "CKA skipped at step %s: a rank received no batches.",
                        step,
                    )
                return
            count = torch.tensor([float(n_total)], device=self.device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            n_total = int(count.item())
            for tensor in (sum_t, sum_s, sum_tt, sum_ss, sum_ts):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if not is_main_process() or n_total == 0 or sum_t is None:
            return

        mean_t = sum_t / n_total
        mean_s = sum_s / n_total
        cov_tt = sum_tt - n_total * torch.outer(mean_t, mean_t)
        cov_ss = sum_ss - n_total * torch.outer(mean_s, mean_s)
        cov_ts = sum_ts - n_total * torch.outer(mean_t, mean_s)

        numerator = cov_ts.pow(2).sum()
        denominator = cov_tt.norm() * cov_ss.norm()
        cka = float((numerator / denominator.clamp_min(self.eps)).item())

        if self.logger:
            self.logger.info(
                "Linear CKA (teacher↔student, CLS) @ step %s: %.4f | N=%d",
                step, cka, n_total,
            )
        if self.tracker is not None:
            self.tracker.log({"eval/linear_cka_cls": cka}, step=step)

    @staticmethod
    def _clean_state_dict(state_dict):
        """Strip ``_orig_mod.`` prefix injected by ``torch.compile``."""
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    def _save_checkpoint(self, step):
        student_state = self._clean_state_dict(_unwrap_ddp(self.student).state_dict())
        payload = {
            "step": step,
            "student_config": self.student_config,
            "student_state": student_state,
            "optimizer_state": {name: optimizer.state_dict() for name, optimizer in self.optimizers},
            "args": vars(self.args),
        }
        if self.scaler.is_enabled():
            payload["scaler_state"] = self.scaler.state_dict()
        if self.args.save_projector and self.projector is not None:
            projector_state = self._clean_state_dict(_unwrap_ddp(self.projector).state_dict())
            payload["projector_state"] = projector_state
        path = os.path.join(self.args.output_dir, f"distill_step_{step}.pt")
        torch.save(payload, path)

    def train(self, dataloader, max_steps, sampler):
        self.student.train()
        if self.projector is not None:
            self.projector.train()
        self.teacher.eval()

        if sampler is not None:
            sampler.set_epoch(0)
        data_iter = iter(dataloader)
        epoch = 0

        # LR schedule
        schedulers = []
        if self.args.lr_schedule != "none":
            warmup = self.args.lr_warmup_steps
            if warmup <= 0:
                warmup = max(int(0.1 * max_steps), 1)
            for _, optimizer in self.optimizers:
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda s: _build_cosine_schedule(
                        s + 1, max_steps, warmup, self.args.min_lr_ratio
                    ),
                    last_epoch=-1,
                )
                schedulers.append(scheduler)

        for step in range(1, max_steps + 1):
            self._current_step = step
            # ── Cyclic data iteration ──
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            if self.pixel_normalize is not None:
                pixel_values = self.pixel_normalize(pixel_values)

            # ── Forward ──
            for _, optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            teacher_cls, teacher_tokens = self._forward_teacher(pixel_values)
            student_cls, student_tokens = self._forward_student(pixel_values)
            loss, loss_metrics = self._compute_loss(
                student_cls,
                student_tokens,
                teacher_cls,
                teacher_tokens,
            )

            # ── NaN guard ──
            if not torch.isfinite(loss):
                msg = f"Non-finite loss at step {step}: {loss}"
                if self.args.abort_on_nan:
                    raise ValueError(msg)
                self.logger.warning(msg)
                continue

            # ── Backward + step ──
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm = None
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                if self.scaler.is_enabled():
                    for _, optimizer in self.optimizers:
                        self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.clip_params, self.args.max_grad_norm)
            if self.scaler.is_enabled():
                for _, optimizer in self.optimizers:
                    self.scaler.step(optimizer)
                self.scaler.update()
            else:
                for _, optimizer in self.optimizers:
                    optimizer.step()
            for scheduler in schedulers:
                scheduler.step()

            current_lrs = {}
            for optimizer_name, optimizer in self.optimizers:
                current_lrs[f"train/lr_{optimizer_name}"] = optimizer.param_groups[0]["lr"]
            if self.optimizers:
                current_lrs["train/lr_student"] = self.optimizers[0][1].param_groups[0]["lr"]
            if grad_norm is not None:
                loss_metrics["train/grad_norm"] = float(grad_norm)
            if self.scaler.is_enabled():
                loss_metrics["train/grad_scale"] = float(self.scaler.get_scale())
            loss_metrics.update(current_lrs)
            reduced_metrics = self._reduce_metrics_for_logging(loss_metrics)

            if (
                is_main_process()
                and self.tracker is not None
                and self.args.wandb_log_every > 0
                and step % self.args.wandb_log_every == 0
            ):
                self.tracker.log(reduced_metrics, step=step)

            # ── Logging ──
            if step % self.args.log_every == 0:
                loss_val = torch.tensor(
                    float(loss_metrics.get("train/loss_total", loss.detach().item())),
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.args.distributed:
                    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                    loss_val = loss_val / get_world_size()
                if is_main_process():
                    self.logger.info(
                        "Step %s/%s | Loss: %.6f | epoch=%s",
                        step, max_steps, loss_val.item(), epoch,
                    )

            # ── Checkpointing ──
            if (is_main_process() and self.args.save_every
                    and self.args.save_every > 0
                    and step % self.args.save_every == 0):
                self._save_checkpoint(step)

            # ── Linear probe evaluation ──
            if (self.args.eval_iter and self.args.eval_iter > 0
                    and step % self.args.eval_iter == 0):
                if self.linear_probe is not None:
                    self.linear_probe.run(step)
                self._measure_cka(step)
                barrier()
