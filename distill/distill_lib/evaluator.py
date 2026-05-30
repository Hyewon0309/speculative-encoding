"""Linear-probe evaluator for periodic quality assessment during distillation."""

import torch
import torch.nn as nn

from distill_lib.data import _eval_collate_fn, create_dataloader
from distill_lib.encoder import _drop_prefix_tokens, _extract_sequence, _unwrap_ddp
from distill_lib.metrics import compute_classification_metrics
from distill_lib.optim import can_use_fused_adamw
from distill_lib.utils import _build_cosine_schedule, get_autocast


class LinearProbeEvaluator:
    def __init__(self, args, student_encoder, train_dataset, eval_dataset,
                 label_encoder, device, logger, student_prefix_tokens, tracker=None):
        self.args = args
        self.student_encoder = student_encoder
        self.device = device
        self.logger = logger
        self.label_encoder = label_encoder
        self.tracker = tracker
        self.autocast = get_autocast(device, precision=args.precision)
        self.student_prefix_tokens = student_prefix_tokens
        self.pool_method = args.pool_method

        train_batch_size = args.linear_prob_batch_size or args.batch_size
        eval_batch_size = args.linear_prob_eval_batch_size or args.batch_size
        self.train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            sampler=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=True,
            collate_fn=_eval_collate_fn,
        )
        self.eval_loader = create_dataloader(
            dataset=eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=False,
            collate_fn=_eval_collate_fn,
        )

        self.num_classes = len(label_encoder.label_to_id)
        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        encoder = _unwrap_ddp(self.student_encoder)
        was_training = encoder.training
        encoder.eval()
        batch = next(iter(self.train_loader))
        pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
        with torch.no_grad():
            with self.autocast:
                features = self._extract_features(encoder, pixel_values)
        if was_training:
            encoder.train()
        return features.size(-1)

    def _extract_features(self, encoder, pixel_values):
        """Forward student -> drop prefix -> pool -> (B, dim) feature vector.

        Caller owns the autocast / grad context so the same extraction logic can
        be used safely for both classifier training (`no_grad`) and eval
        (`inference_mode`).
        """
        outputs = encoder(pixel_values)
        seq = _extract_sequence(outputs)
        if self.pool_method == "cls":
            return seq[:, 0]
        seq = _drop_prefix_tokens(seq, self.student_prefix_tokens)
        return seq.mean(dim=1)

    def _next_batch(self, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(self.train_loader)
            return next(iterator), iterator

    def run(self, step: int):
        """Freeze student -> train linear head -> evaluate -> restore student."""
        encoder = _unwrap_ddp(self.student_encoder)

        # 1. Save training state
        was_training = encoder.training
        grad_states = [p.requires_grad for p in encoder.parameters()]

        # 2. Freeze student
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        # 3. Create fresh classifier
        classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        classifier.train()
        adamw_groups = [{"params": list(classifier.parameters())}]
        adamw_kwargs = {
            "lr": self.args.linear_prob_lr,
            "weight_decay": self.args.linear_prob_weight_decay,
        }
        if can_use_fused_adamw(adamw_groups):
            adamw_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(adamw_groups, **adamw_kwargs)

        # 4. Optional cosine schedule
        scheduler = None
        if self.args.linear_prob_schedule != "none":
            warmup_steps = self.args.linear_prob_warmup_steps
            if warmup_steps <= 0:
                warmup_steps = max(int(0.1 * self.args.linear_prob_iter), 1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda s: _build_cosine_schedule(
                    s, self.args.linear_prob_iter, warmup_steps,
                    self.args.linear_prob_min_lr_ratio,
                ),
            )
        criterion = nn.CrossEntropyLoss()

        # 5. Train classifier
        train_iter = iter(self.train_loader)
        for _ in range(self.args.linear_prob_iter):
            batch, train_iter = self._next_batch(train_iter)
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with self.autocast:
                    features = self._extract_features(encoder, pixel_values)
            features = features.float()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), self.args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # 6. Evaluate
        classifier.eval()
        logits_list, labels_list = [], []
        with torch.inference_mode():
            for batch in self.eval_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                with self.autocast:
                    features = self._extract_features(encoder, pixel_values)
                features = features.float()
                logits = classifier(features)
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        metrics = compute_classification_metrics(
            all_logits, all_labels,
            label_names=self.label_encoder.id_to_label,
        )
        if self.logger and metrics:
            self.logger.info("Linear probe @ step %s | metrics=%s", step, metrics)
        if self.tracker and metrics:
            self.tracker.log(
                {f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step,
            )

        # 7. Restore student training state
        for p, requires_grad in zip(encoder.parameters(), grad_states):
            p.requires_grad = requires_grad
        if was_training:
            encoder.train()

        return metrics
