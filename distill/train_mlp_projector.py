"""Train a student→teacher MLP projector (MSE, SiLU).

Both teacher and student are frozen. Only the projector's weights move.
CLS features are compared because CONCH v1.5 with projection returns a
single pooled token — and in general, student/teacher patch grids can
differ. The projector learns ``student_dim → teacher_dim``.

Launch example (see ``train_mlp_projector_conch.sh``)::

    torchrun --nproc_per_node=8 train_mlp_projector.py \
        --data_dir ${DISTILL_DATA_DIR} \
        --teacher_model conchv15 \
        --student_ckpt_path /path/to/distill_step_10000.pt \
        --output_path /path/to/mlp_projector.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from distill_lib.data import (
    ImageFolderDataset,
    _distill_collate_fn,
    create_dataloader,
    load_image_paths_cache,
    resolve_image_cache_path,
    save_image_paths_cache,
    scan_image_paths,
)
from distill_lib.distributed import (
    barrier,
    cleanup,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
)
from distill_lib.encoder import (
    GPUPixelNormalize,
    _drop_prefix_tokens,
    _extract_sequence,
    _infer_image_prefix_tokens,
    _unwrap_ddp,
)
from distill_lib.mlp_projector import MLPProjector, save_mlp_projector
from distill_lib.student import load_student_from_checkpoint
from distill_lib.teacher import load_teacher
from distill_lib.utils import (
    WandbTracker,
    _build_cosine_schedule,
    _configure_warnings,
    get_autocast,
    get_grad_scaler,
    get_logger,
    set_seed,
    timed_stage,
)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a student→teacher MLP projector")

    # Teacher
    teacher_choices = [
        "uni", "biomedclip", "openaiclip", "plip", "medsiglip", "conch",
        "conchv15", "conch_v15", "conchv1_5",
    ]
    p.add_argument("--teacher_model", type=str, default="conchv15",
                   choices=teacher_choices)
    p.add_argument("--teacher_model_path", type=str, default=None)
    p.add_argument("--teacher_model_name", type=str, default=None)
    p.add_argument("--conchv15_use_projection", dest="conchv15_use_projection",
                   action="store_true", default=True)
    p.add_argument("--no_conchv15_use_projection", dest="conchv15_use_projection",
                   action="store_false")

    # Student: must be a distilled checkpoint (frozen)
    p.add_argument("--student_ckpt_path", type=str, required=True,
                   help="Path to a distillation checkpoint. Carries its own "
                        "student_config, so architecture flags are unnecessary.")

    # MLP architecture
    p.add_argument("--mlp_hidden_dim", type=int, default=None,
                   help="Hidden width. Default: teacher dim.")
    p.add_argument("--mlp_num_hidden_layers", type=int, default=2,
                   help="Number of hidden Linear+SiLU blocks. "
                        "0 collapses to a single linear map.")
    p.add_argument("--mlp_activation", type=str, default="silu",
                   choices=["silu", "gelu", "relu"])

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--no_persistent_workers", dest="persistent_workers",
                   action="store_false")
    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--image_index_workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--refresh_image_cache", action="store_true")

    # Training
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["none", "cosine"])
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="fp16",
                   choices=["bf16", "fp16", "fp32"])

    # IO
    p.add_argument("--output_path", type=str, required=True,
                   help="Destination file for the trained MLP checkpoint.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=0,
                   help="If >0, write intermediate MLP checkpoints every N steps "
                        "as ``{output_path}.step{N}.pt``.")

    # W&B
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_log_every", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default="speculative-encoding-mlp-projector")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)

    # Misc
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--compile_mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--suppress_warnings", action="store_true", default=True)
    p.add_argument("--show_warnings", dest="suppress_warnings", action="store_false")
    p.add_argument("--abort_on_nan", action="store_true", default=True)
    p.add_argument("--no_abort_on_nan", dest="abort_on_nan", action="store_false")

    args = p.parse_args()
    # Fields the shared helpers expect but we don't expose here.
    args.image_cache_dir = os.path.join(os.path.dirname(args.output_path), "_image_cache")
    if not args.wandb_run_name:
        args.wandb_run_name = os.path.splitext(os.path.basename(args.output_path))[0]
    return args


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_cls(module, pixel_values, prefix_tokens):
    """Run a frozen encoder and return its CLS-style feature ``(B, D)``.

    Teacher (CONCH v1.5 with projection) returns ``(B, 1, 768)`` — index 0.
    Student returns ``(B, 1+N, 384)`` — index 0 is the CLS token.
    """
    del prefix_tokens  # CLS is always token 0; kept for API symmetry.
    tokens = _extract_sequence(module(pixel_values))
    return tokens[:, 0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _configure_warnings(args)

    init_distributed_mode(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed, getattr(args, "rank", 0))

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(
        "mlp_projector",
        log_path=(
            os.path.splitext(args.output_path)[0] + ".log"
            if is_main_process() else None
        ),
    )
    tracker = WandbTracker(enabled=args.wandb)
    tracker.init(args=args, output_dir=output_dir)

    # ── Load teacher (frozen) ──
    with timed_stage(logger, "load_teacher"):
        teacher, _, distill_preprocess, teacher_meta = load_teacher(args)
    teacher_prefix = _infer_image_prefix_tokens(teacher)
    teacher = teacher.to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    pixel_normalize = GPUPixelNormalize(
        teacher_meta.get("mean", (0.485, 0.456, 0.406)),
        teacher_meta.get("std", (0.229, 0.224, 0.225)),
    ).to(device)

    # ── Load student from distilled checkpoint (frozen) ──
    with timed_stage(logger, "load_student"):
        student_raw, student_config = load_student_from_checkpoint(
            args.student_ckpt_path, map_location="cpu", return_config=True,
        )
    student_prefix = _infer_image_prefix_tokens(student_raw)
    student = student_raw.to(device)
    for param in student.parameters():
        param.requires_grad = False
    student.eval()

    if is_main_process():
        logger.info("Loaded student from %s", args.student_ckpt_path)
        logger.info("Student config: %s", student_config)

    # ── Probe dims with a single forward ──
    image_cache_path = resolve_image_cache_path(args.data_dir, args.image_cache_dir)
    with timed_stage(logger, "prepare_train_image_index"):
        if is_main_process():
            if args.refresh_image_cache or not os.path.exists(image_cache_path):
                paths = scan_image_paths(args.data_dir, num_workers=args.image_index_workers)
                save_image_paths_cache(paths, image_cache_path)
                logger.info("Saved image cache %s (%d images)", image_cache_path, len(paths))
            else:
                logger.info("Reusing image cache %s", image_cache_path)
        barrier()
        distill_paths = load_image_paths_cache(image_cache_path)

    distill_ds = ImageFolderDataset(
        data_dir=args.data_dir,
        preprocess=distill_preprocess,
        max_samples=args.max_train_samples,
        seed=args.seed,
        paths=distill_paths,
    )
    if is_main_process():
        logger.info("Distillation dataset: %d images", len(distill_ds))

    sampler = DistributedSampler(distill_ds, shuffle=True) if args.distributed else None
    train_loader = create_dataloader(
        dataset=distill_ds,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last,
        collate_fn=_distill_collate_fn,
    )

    autocast = get_autocast(device, precision=args.precision)
    with timed_stage(logger, "probe_dims"):
        probe_batch = next(iter(train_loader))
        probe_pixels = pixel_normalize(probe_batch["pixel_values"].to(device, non_blocking=True))
        with torch.inference_mode(), autocast:
            t_cls = _extract_cls(teacher, probe_pixels, teacher_prefix)
            s_cls = _extract_cls(student, probe_pixels, student_prefix)
    teacher_dim = int(t_cls.size(-1))
    student_dim = int(s_cls.size(-1))
    if is_main_process():
        logger.info("Dims: student=%d → teacher=%d", student_dim, teacher_dim)

    # ── Build MLP projector (trainable) ──
    projector = MLPProjector(
        in_dim=student_dim,
        out_dim=teacher_dim,
        hidden_dim=args.mlp_hidden_dim,
        num_hidden_layers=args.mlp_num_hidden_layers,
        activation=args.mlp_activation,
    ).to(device)
    if is_main_process():
        n_params = sum(p.numel() for p in projector.parameters())
        logger.info("MLP projector: %s (%d params)", projector.config(), n_params)

    if args.compile:
        if is_main_process():
            logger.info("Applying torch.compile (mode=%s)", args.compile_mode)
        teacher = torch.compile(teacher, mode=args.compile_mode)
        student = torch.compile(student, mode=args.compile_mode)
        projector = torch.compile(projector, mode=args.compile_mode)

    projector_ddp = projector
    if args.distributed:
        projector_ddp = DDP(
            projector,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = torch.optim.AdamW(
        projector_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_schedule != "none":
        warmup = args.lr_warmup_steps if args.lr_warmup_steps > 0 else max(int(0.1 * args.max_steps), 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: _build_cosine_schedule(
                s + 1, args.max_steps, warmup, args.min_lr_ratio,
            ),
        )
    scaler = get_grad_scaler(device, precision=args.precision)

    # ── Training loop ──
    projector_ddp.train()
    teacher.eval()
    student.eval()

    if sampler is not None:
        sampler.set_epoch(0)
    data_iter = iter(train_loader)
    epoch = 0

    last_loss = None
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_values = pixel_normalize(pixel_values)

        # Frozen features — no grad graph beyond the projector.
        with torch.inference_mode(), autocast:
            teacher_cls = _extract_cls(teacher, pixel_values, teacher_prefix)
            student_cls = _extract_cls(student, pixel_values, student_prefix)
        teacher_cls = teacher_cls.clone()
        student_cls = student_cls.clone()

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            projected = projector_ddp(student_cls)
            loss = F.mse_loss(projected.float(), teacher_cls.float())

        if not torch.isfinite(loss):
            msg = f"Non-finite loss at step {step}: {loss}"
            if args.abort_on_nan:
                raise ValueError(msg)
            if is_main_process():
                logger.warning(msg)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = None
        if args.max_grad_norm and args.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                projector_ddp.parameters(), args.max_grad_norm,
            )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.detach().float()
        if args.distributed:
            torch.distributed.all_reduce(loss_val, op=torch.distributed.ReduceOp.SUM)
            loss_val = loss_val / get_world_size()
        loss_val = float(loss_val)
        last_loss = loss_val

        with torch.no_grad():
            cos = F.cosine_similarity(projected.float(), teacher_cls.float(), dim=-1).mean()

        if (
            is_main_process()
            and args.wandb
            and args.wandb_log_every > 0
            and step % args.wandb_log_every == 0
        ):
            tracker.log(
                {
                    "train/loss_mse": loss_val,
                    "train/cosine": float(cos),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "train/grad_scale": (
                        float(scaler.get_scale()) if scaler.is_enabled() else None
                    ),
                },
                step=step,
            )

        if step % args.log_every == 0 and is_main_process():
            logger.info(
                "Step %d/%d | mse=%.6f | cos=%.4f | lr=%.2e | epoch=%d",
                step, args.max_steps, loss_val, float(cos),
                optimizer.param_groups[0]["lr"], epoch,
            )

        if (
            args.save_every > 0
            and step % args.save_every == 0
            and step != args.max_steps
            and is_main_process()
        ):
            intermediate_path = (
                f"{os.path.splitext(args.output_path)[0]}.step{step}.pt"
            )
            save_mlp_projector(
                _unwrap_ddp(projector_ddp),
                intermediate_path,
                meta=_build_meta(args, student_config, teacher_meta, step, loss_val),
            )
            logger.info("Saved intermediate MLP checkpoint → %s", intermediate_path)

    # ── Final save ──
    if is_main_process():
        save_mlp_projector(
            _unwrap_ddp(projector_ddp),
            args.output_path,
            meta=_build_meta(args, student_config, teacher_meta, args.max_steps, last_loss),
        )
        logger.info("Saved final MLP projector → %s", args.output_path)
        # Side-by-side JSON with the human-readable config.
        config_json_path = os.path.splitext(args.output_path)[0] + ".json"
        with open(config_json_path, "w") as f:
            json.dump(
                {
                    "mlp_config": _unwrap_ddp(projector_ddp).config(),
                    "meta": _build_meta(args, student_config, teacher_meta, args.max_steps, last_loss),
                },
                f, indent=2,
            )

    tracker.finish()
    barrier()
    cleanup()


def _build_meta(args, student_config, teacher_meta, step, loss):
    return {
        "teacher_model": args.teacher_model,
        "teacher_embed_dim": int(teacher_meta.get("embed_dim", 0)),
        "teacher_img_size": int(teacher_meta.get("img_size", 0)),
        "student_ckpt_path": os.path.abspath(args.student_ckpt_path),
        "student_config": dict(student_config),
        "step": int(step),
        "final_loss_mse": float(loss) if loss is not None else None,
        "data_dir": os.path.abspath(args.data_dir),
    }


if __name__ == "__main__":
    main()
