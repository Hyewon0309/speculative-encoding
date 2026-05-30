"""Main orchestration: ties all modules together for the distillation pipeline."""

import inspect
import json
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from distill_lib.args import parse_args
from distill_lib.data import (
    ImageFolderDataset,
    _distill_collate_fn,
    build_eval_transform,
    create_dataloader,
    load_image_paths_cache,
    load_nctcrche100k,
    resolve_image_cache_path,
    save_image_paths_cache,
    scan_image_paths,
)
from distill_lib.distributed import barrier, cleanup, init_distributed_mode, is_main_process
from distill_lib.encoder import (
    GPUPixelNormalize,
    ImageEncoderWrapper,
    _drop_prefix_tokens,
    _extract_sequence,
    _infer_image_prefix_tokens,
)
from distill_lib.evaluator import LinearProbeEvaluator
from distill_lib.projector import DistillationProjector
from distill_lib.student import build_student_config, initialize_student_from_checkpoint, load_student
from distill_lib.teacher import load_teacher
from distill_lib.trainer import Distiller
from distill_lib.utils import (
    _configure_warnings,
    WandbTracker,
    get_autocast,
    get_logger,
    set_seed,
    timed_stage,
)


def _format_image_count(count: int) -> str:
    if count >= 999_500:
        return f"{count / 1_000_000:.1f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return str(count)


def main():
    # ── 1. Parse CLI arguments ──
    args = parse_args()
    _configure_warnings(args)

    # ── 2. Distributed setup ──
    init_distributed_mode(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed, getattr(args, "rank", 0))

    # ── 3. Skip runs that would reuse an existing output directory ──
    skip_existing_output = False
    if is_main_process():
        skip_existing_output = os.path.exists(args.output_dir)
    if args.distributed:
        skip_flag = torch.tensor([int(skip_existing_output)], device=device)
        torch.distributed.broadcast(skip_flag, src=0)
        skip_existing_output = bool(skip_flag.item())
    if skip_existing_output:
        logger = get_logger("distill")
        if is_main_process():
            logger.warning(
                "\033[31mOutput directory already exists. Skipping run: %s\033[0m",
                args.output_dir,
            )
        cleanup()
        return

    # ── 4. Create output directory and logger ──
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=False)

    logger = get_logger(
        "distill",
        log_path=os.path.join(args.output_dir, "distill.log") if is_main_process() else None,
    )
    tracker = WandbTracker(enabled=args.wandb)
    tracker.init(args=args, output_dir=args.output_dir)

    # ── 5. Load teacher (frozen) ──
    with timed_stage(logger, "load_teacher"):
        teacher, teacher_preprocess, distill_preprocess, teacher_meta = load_teacher(args)
    teacher_prefix = _infer_image_prefix_tokens(teacher)
    teacher = teacher.to(device)

    # GPU-side uint8 → float32 + normalize. The dataloader emits uint8 frames
    # so that pin-memory / PCIe bandwidth scales with raw bytes; this module
    # handles the `.float() / 255 → (x - mean) / std` step on-device.
    pixel_mean = teacher_meta.get("mean", (0.485, 0.456, 0.406))
    pixel_std = teacher_meta.get("std", (0.229, 0.224, 0.225))
    pixel_normalize = GPUPixelNormalize(pixel_mean, pixel_std).to(device)

    # Auto-set student spatial config from teacher if not explicitly provided
    if args.student_img_size is None:
        args.student_img_size = teacher_meta["img_size"]
    if args.student_patch_size is None:
        args.student_patch_size = teacher_meta["patch_size"]

    # ── 6. Build student config (after teacher spatial resolution is known) ──
    with timed_stage(logger, "build_student"):
        student_config = build_student_config(args)

    if is_main_process():
        with open(os.path.join(args.output_dir, "distill_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.output_dir, "student_config.json"), "w") as f:
            json.dump(student_config, f, indent=2)

    # ── 7. Load student (wrapped so forward() -> forward_features()) ──
    with timed_stage(logger, "load_student"):
        student_raw = load_student(student_config)
        if args.load_model_path:
            initialize_student_from_checkpoint(student_raw, args.load_model_path)
            if is_main_process():
                logger.info("Initialized student weights from %s", args.load_model_path)
    student_prefix = _infer_image_prefix_tokens(student_raw)
    student = ImageEncoderWrapper(student_raw).to(device)

    if is_main_process():
        n_params = sum(p.numel() for p in student_raw.parameters())
        logger.info("Student config: %s  (%s params)", student_config, n_params)

    # ── 8. Load distillation training data (unlabeled image directory) ──
    image_cache_path = resolve_image_cache_path(args.data_dir, args.image_cache_dir)
    with timed_stage(logger, "prepare_train_image_index"):
        if is_main_process():
            cache_exists = os.path.exists(image_cache_path)
            if args.refresh_image_cache or not cache_exists:
                logger.info(
                    "Scanning train image paths with %d workers from %s",
                    args.image_index_workers, args.data_dir,
                )
                distill_paths = scan_image_paths(
                    args.data_dir,
                    num_workers=args.image_index_workers,
                )
                save_image_paths_cache(distill_paths, image_cache_path)
                logger.info(
                    "Saved image path cache: %s | %s images",
                    image_cache_path, _format_image_count(len(distill_paths)),
                )
            else:
                logger.info("Loading cached image index: %s", image_cache_path)
        barrier()
        distill_paths = load_image_paths_cache(image_cache_path)
        if is_main_process():
            logger.info(
                "Loaded image path cache: %s | %s images",
                image_cache_path, _format_image_count(len(distill_paths)),
            )

    with timed_stage(logger, "build_distill_dataset"):
        distill_ds = ImageFolderDataset(
            data_dir=args.data_dir,
            preprocess=distill_preprocess,
            max_samples=args.max_train_samples,
            seed=args.seed,
            paths=distill_paths,
        )
    num_distill_images = len(distill_ds)
    if is_main_process():
        logger.info(
            "Distillation dataset: %s images from %s",
            _format_image_count(num_distill_images), args.data_dir,
        )
        if args.max_train_samples:
            logger.info(
                "  (capped to %s via --max_train_samples)",
                _format_image_count(args.max_train_samples),
            )

    sampler = DistributedSampler(distill_ds, shuffle=True) if args.distributed else None
    with timed_stage(logger, "build_train_loader"):
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

    # CKA loader: fixed train-image subset sharded across ranks via
    # DistributedSampler so all 8 GPUs forward in parallel; statistics are
    # all-reduced at measurement time.
    cka_loader = None
    if args.cka_num_samples and args.cka_num_samples > 0:
        num_cka = min(args.cka_num_samples, len(distill_ds))
        cka_subset = torch.utils.data.Subset(distill_ds, list(range(num_cka)))
        cka_bs = args.cka_batch_size or args.linear_prob_batch_size or args.batch_size
        if args.distributed:
            cka_sampler = DistributedSampler(
                cka_subset, shuffle=False, drop_last=True,
            )
        else:
            cka_sampler = None
        cka_loader = create_dataloader(
            dataset=cka_subset,
            batch_size=cka_bs,
            shuffle=False,
            sampler=cka_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
            drop_last=False,
            collate_fn=_distill_collate_fn,
        )
        if is_main_process():
            logger.info(
                "CKA measurement: %s train images (batch=%s, sharded across ranks)",
                _format_image_count(num_cka), cka_bs,
            )

    # ── 9. Load CRC-100K evaluation data (labeled, for linear probe only) ──
    eval_train_ds, eval_val_ds, label_encoder = None, None, None
    if args.eval_iter and args.eval_iter > 0:
        with timed_stage(logger, "load_eval_dataset"):
            eval_train_ds, eval_val_ds, label_encoder = load_nctcrche100k(
                cache_dir=args.eval_cache_dir,
                dataset_name=args.eval_dataset_name,
                seed=args.seed,
            )
            eval_transform = build_eval_transform(teacher_preprocess, label_encoder)
            eval_train_ds = eval_train_ds.with_transform(eval_transform)
            eval_val_ds = eval_val_ds.with_transform(eval_transform)

    # ── 10. Probe dimensions (one-batch forward) ──
    autocast = get_autocast(device, precision=args.precision)
    with timed_stage(logger, "probe_model_shapes"):
        probe_batch = next(iter(train_loader))
        probe_pixels = probe_batch["pixel_values"].to(device, non_blocking=True)
        probe_pixels = pixel_normalize(probe_pixels)

        with torch.inference_mode():
            with autocast:
                t_out = _extract_sequence(teacher(probe_pixels))
                t_out = _drop_prefix_tokens(t_out, teacher_prefix)
                s_out = _extract_sequence(student(probe_pixels))
                s_out = _drop_prefix_tokens(s_out, student_prefix)

    teacher_seq, teacher_dim = t_out.size(1), t_out.size(2)
    student_seq, student_dim = s_out.size(1), s_out.size(2)
    requires_matching_patch_tokens = any(
        weight > 0 for weight in (
            args.feat_loss_weight,
            args.structure_loss_weight,
            args.cls_structure_loss_weight,
        )
    )
    if requires_matching_patch_tokens:
        assert teacher_seq == student_seq, (
            f"Token count mismatch: teacher={teacher_seq}, student={student_seq}. "
            "The active token-level distillation losses require matching patch grids, "
            "so teacher and student must use the same patch_size."
        )

    if is_main_process():
        logger.info(
            "Teacher: %s tokens x %sd | Student: %s tokens x %sd",
            teacher_seq, teacher_dim, student_seq, student_dim,
        )
        if teacher_seq != student_seq and not requires_matching_patch_tokens:
            logger.warning(
                "Teacher/student patch token counts differ (%s vs %s), but proceeding "
                "because only CLS-compatible losses are active.",
                teacher_seq, student_seq,
            )

    # ── 11. Build projector only for legacy direct-regression losses ──
    use_projector = any(
        weight > 0 for weight in (
            args.feat_loss_weight,
            args.cls_loss_weight,
            args.cosine_loss_weight,
        )
    )
    projector = None
    if use_projector:
        projector = DistillationProjector(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        ).to(device)
    if args.new_loss and is_main_process():
        logger.info(
            "Using pairwise L2-distance MSE objective (sqrt(d) normalized) for global CLS."
        )

    # ── 11. Full distillation — all student params trainable ──
    for p in student.parameters():
        p.requires_grad = True

    # ── 11b. torch.compile (before DDP) ──
    if args.compile:
        logger.info("Applying torch.compile (mode=%s)", args.compile_mode)
        teacher = torch.compile(teacher, mode=args.compile_mode)
        student = torch.compile(student, mode=args.compile_mode)
        if projector is not None:
            projector = torch.compile(projector, mode=args.compile_mode)

    # ── 12. Wrap with DDP (if distributed) ──
    if args.distributed:
        ddp_kwargs = {
            "device_ids": [args.local_rank],
            "output_device": args.local_rank,
            "broadcast_buffers": False,
            "find_unused_parameters": args.ddp_find_unused_parameters,
        }
        if "static_graph" in inspect.signature(DDP).parameters:
            ddp_kwargs["static_graph"] = not args.ddp_find_unused_parameters
        student = DDP(
            student,
            **ddp_kwargs,
        )
        if projector is not None:
            projector = DDP(
                projector,
                **ddp_kwargs,
            )

    # ── 13. Build linear probe evaluator (main process only, uses CRC-100K) ──
    linear_probe = None
    if args.eval_iter and args.eval_iter > 0:
        if is_main_process():
            linear_probe = LinearProbeEvaluator(
                args=args,
                student_encoder=student,
                train_dataset=eval_train_ds,
                eval_dataset=eval_val_ds,
                label_encoder=label_encoder,
                device=device,
                logger=logger,
                student_prefix_tokens=student_prefix,
                tracker=tracker,
            )

    # ── 14. Build distiller and compute max_steps ──
    distiller = Distiller(
        args=args,
        teacher=teacher,
        student=student,
        projector=projector,
        device=device,
        logger=logger,
        linear_probe=linear_probe,
        teacher_prefix_tokens=teacher_prefix,
        student_prefix_tokens=student_prefix,
        student_config=student_config,
        tracker=tracker,
        pixel_normalize=pixel_normalize,
        cka_loader=cka_loader,
    )

    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = args.epochs * len(train_loader)

    # ── 15. Train ──
    distiller.train(train_loader, max_steps, sampler)

    # ── 16. Final checkpoint ──
    if is_main_process() and args.save_every and args.save_every > 0:
        if max_steps % args.save_every != 0:
            distiller._save_checkpoint(max_steps)

    # ── 17. Cleanup ──
    tracker.finish()
    barrier()
    cleanup()
