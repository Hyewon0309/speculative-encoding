"""CLI argument parsing for the distillation script."""

import argparse
import json
import os
import re


def _infer_teacher_patch_size(args):
    if args.teacher_model == "uni":
        return 14
    if args.teacher_model in {"conch", "conchv15", "conch_v15", "conchv1_5"}:
        return 16
    if args.teacher_model in {"virchow", "prism"}:
        return 14
    if args.teacher_model in {
        "provgigapath", "prov_gigapath", "prov-gigapath", "gigapath",
    }:
        return 16
    if args.teacher_model == "openaiclip":
        arch_name = args.teacher_model_path or "ViT-L-14"
        match = re.search(r"-(\d+)$", arch_name)
        if match:
            return int(match.group(1))
        return None
    if args.teacher_model == "biomedclip" and args.teacher_model_path:
        config_path = os.path.join(args.teacher_model_path, "open_clip_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("model_cfg", {}).get("vision_cfg", {}).get("patch_size")
    return None


def _resolve_student_patch_size(args):
    if args.student_patch_size is not None:
        return args.student_patch_size
    return _infer_teacher_patch_size(args)


def _build_config_tag(args) -> str:
    student_patch_size = _resolve_student_patch_size(args)
    config_bits = [
        args.teacher_model,
        f"dim_{args.student_dim}",
        f"layer_{args.student_depth}",
        f"head_{args.student_heads}",
    ]
    if student_patch_size is not None:
        config_bits.append(f"patch_{student_patch_size}")
    if args.student_mlp_ratio != 4.0:
        config_bits.append(f"mlp_{str(args.student_mlp_ratio).replace('.', 'p')}")
    if args.student_act != "gelu":
        config_bits.append(args.student_act)
    return "_".join(config_bits)


def _build_run_tag(args, config_tag: str) -> str:
    run_parts = [config_tag]

    loss_abbrevs = []
    if args.feat_loss_weight > 0:
        loss_abbrevs.append("feat")
    if args.cls_loss_weight > 0:
        loss_abbrevs.append("cls")
    if args.cosine_loss_weight > 0:
        loss_abbrevs.append("cos")
    if args.structure_loss_weight > 0:
        loss_abbrevs.append("str")
    if args.cls_structure_loss_weight > 0:
        loss_abbrevs.append("cstr")
    if args.new_loss:
        loss_abbrevs.append("pl2")
    elif args.global_structure_loss_weight > 0:
        loss_abbrevs.append("gstr")
    if args.only_affinity:
        loss_abbrevs.append("affOnly")
    elif args.only_distance:
        loss_abbrevs.append("distOnly")
    run_parts.append("L_" + "+".join(loss_abbrevs))

    temp_strs = [str(t).replace('.', 'p') for t in args.structure_temperature]
    run_parts.append("t_" + "+".join(temp_strs))

    if args.optimizer != "adamw":
        run_parts.append(args.optimizer)

    run_parts.append(f"prec_{args.precision}")
    run_parts.append(f"lr_{args.lr:.0e}".replace("+", ""))
    run_parts.append(f"wd_{args.weight_decay:.0e}".replace("+", ""))
    return "_".join(run_parts)


def parse_args():
    p = argparse.ArgumentParser(description="Distill UNI vision encoder")

    # ── Teacher ──
    teacher_choices = [
        "uni", "biomedclip", "openaiclip", "plip", "medsiglip", "conch",
        "conchv15", "conch_v15", "conchv1_5",
        "virchow", "prism",
        "provgigapath", "prov_gigapath", "prov-gigapath", "gigapath",
    ]
    p.add_argument("--teacher_model", type=str, default="uni",
                    choices=teacher_choices,
                    help="Teacher encoder name.")
    p.add_argument("--teacher_model_path", type=str, default=None,
                    help="Local model weights path or HF model identifier. "
                         "Required for biomedclip, plip. For openaiclip, use as "
                         "architecture name (e.g. 'ViT-L-14'). For conchv15, this "
                         "may be a checkpoint file, a directory containing "
                         "'pytorch_model_vision.bin', or an HF repo id.")
    p.add_argument("--teacher_model_name", type=str, default=None,
                    help="Model config name for open_clip registration "
                         "(used by biomedclip, e.g. 'biomedclip_local').")
    p.add_argument("--conchv15_use_projection", dest="conchv15_use_projection",
                    action="store_true", default=True,
                    help="For --teacher_model conchv15: include the 1024→768 "
                         "attentional-pool contrast projection (default: on). "
                         "Teacher output becomes (B, 1, 768).")
    p.add_argument("--no_conchv15_use_projection", dest="conchv15_use_projection",
                    action="store_false",
                    help="Skip the CONCH v1.5 contrast projection and use the "
                         "bare ViT-L/16 trunk (teacher output (B, 1+N, 1024)).")

    # ── Student architecture ──
    p.add_argument("--student_img_size", type=int, default=None,
                    help="Student image size. Auto-derived from teacher if None.")
    p.add_argument("--student_patch_size", type=int, default=None,
                    help="Student patch size. Auto-derived from teacher if None.")
    p.add_argument("--student_dim", type=int, default=384,
                    help="Student embed_dim.")
    p.add_argument("--student_depth", type=int, default=12,
                    help="Number of Transformer layers in the student.")
    p.add_argument("--student_heads", type=int, default=6,
                    help="Number of attention heads in the student.")
    p.add_argument("--student_act", type=str, default="gelu",
                    choices=["gelu", "silu", "swiglu"],
                    help="Activation function for the student MLP. "
                         "'swiglu' uses SwiGLUPacked MLP with SiLU activation.")
    p.add_argument("--student_mlp_ratio", type=float, default=4.0,
                    help="Student FFN expansion ratio.")

    # ── Distillation data (unlabeled image directory) ──
    p.add_argument("--data_dir", type=str, required=True,
                    help="Directory of JPG/PNG images for distillation training. "
                         "Recursively scans for *.jpg, *.jpeg, *.png files.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--image_index_workers", type=int, default=min(32, os.cpu_count() or 8),
                    help="Thread count for rank-0 train image index construction.")
    p.add_argument("--refresh_image_cache", dest="refresh_image_cache", action="store_true",
                    help="Force rebuilding the cached train image path list.")
    p.add_argument("--no_refresh_image_cache", dest="refresh_image_cache", action="store_false",
                    help="Reuse the cached train image path list when present.")

    # ── Evaluation data (CRC-100K from HuggingFace, for linear probe only) ──
    p.add_argument("--eval_dataset_name", type=str, default="DykeF/NCTCRCHE100K")
    p.add_argument(
        "--eval_cache_dir",
        type=str,
        default=os.environ.get("EVAL_CACHE_DIR", ""),
        help="HuggingFace datasets cache for the CRC-100K linear-probe eval set. "
             "Defaults to $EVAL_CACHE_DIR (configs/paths.json).",
    )

    # ── Training ──
    p.add_argument("--max_steps", type=int, default=None,
                    help="Step-based training. Overrides --epochs when set.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "muon"],
                    help="Optimizer for student training.")
    p.add_argument("--load_model_path", type=str, default=None,
                    help="Initialize the student weights from a distillation checkpoint. "
                         "Only model weights are loaded; optimizer, scaler, and step are not resumed.")
    p.add_argument("--projector_lr", type=float, default=None,
                    help="Separate LR for projectors. Defaults to --lr.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--muon_momentum", type=float, default=0.95,
                    help="Momentum used by torch.optim.Muon.")
    p.add_argument("--muon_ns_steps", type=int, default=5,
                    help="Newton-Schulz iterations used by torch.optim.Muon.")
    p.add_argument("--muon_eps", type=float, default=1e-7,
                    help="Epsilon used by torch.optim.Muon.")
    p.add_argument("--muon_nesterov", dest="muon_nesterov", action="store_true",
                    help="Enable Nesterov momentum for torch.optim.Muon.")
    p.add_argument("--no_muon_nesterov", dest="muon_nesterov", action="store_false",
                    help="Disable Nesterov momentum for torch.optim.Muon.")
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--lr_warmup_steps", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"],
                    help="Mixed-precision mode: bf16, fp16, or fp32 (no autocast).")

    # ── Loss weights ──
    p.add_argument("--feat_loss_weight", type=float, default=0.0,
                    help="Weight for per-token feature MSE loss.")
    p.add_argument("--cls_loss_weight", type=float, default=0.0,
                    help="Weight for mean-pooled MSE loss.")
    p.add_argument("--cosine_loss_weight", type=float, default=0.0,
                    help="Weight for cosine similarity loss (0 = disabled).")
    p.add_argument("--structure_loss_weight", type=float, default=1.0,
                    help="Weight for intra-image patch-structure matching loss.")
    p.add_argument("--cls_structure_loss_weight", type=float, default=1.0,
                    help="Weight for CLS-to-patch similarity matching loss.")
    p.add_argument("--global_structure_loss_weight", type=float, default=0.25,
                    help="Weight for inter-image CLS similarity matching loss.")
    p.add_argument("--structure_loss_type", type=str, default="kl",
                    choices=["mse", "smooth_l1", "kl"],
                    help="Loss used to compare similarity structures.")
    p.add_argument("--structure_temperature", type=float, nargs='+', default=[1.0],
                    help="Temperature(s) applied to structural similarity targets. "
                         "Multiple values enable multi-scale relational matching.")
    p.add_argument("--new_loss", dest="new_loss", action="store_true",
                    help="Replace the global CLS relational loss with the "
                         "pairwise L2-distance MSE objective (sqrt(d) normalized).")
    p.add_argument("--no_new_loss", dest="new_loss", action="store_false",
                    help="Disable the pairwise L2-distance MSE objective.")
    p.add_argument("--new_loss_weight", type=float, default=1.0,
                    help="Weight applied to the pairwise L2-distance MSE loss.")
    p.add_argument("--only_affinity", dest="only_affinity", action="store_true",
                    help="Ablation: keep only the affinity (gram) component of "
                         "structure / cls_structure / global_structure losses.")
    p.add_argument("--only_distance", dest="only_distance", action="store_true",
                    help="Ablation: keep only the distance component of "
                         "structure / cls_structure / global_structure losses.")

    # ── Logging & checkpointing ──
    p.add_argument("--output_base", type=str, default="./outputs/distill",
                    help="Base output directory. A subdirectory named "
                        "dim_{d}_layer_{l}_head_{h} is auto-created.")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wandb_log_every", type=int, default=10,
                    help="Log training metrics to W&B every N steps.")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--save_projector", dest="save_projector", action="store_true",
                    help="Save the projector weights in checkpoints (default: True).")
    p.add_argument("--no_save_projector", dest="save_projector", action="store_false",
                    help="Exclude projector weights from checkpoints.")

    # ── Linear probing evaluation ──
    p.add_argument("--eval_iter", type=int, default=500,
                    help="Run linear probe every N steps. 0 = disabled.")
    p.add_argument("--linear_prob_iter", type=int, default=500)
    p.add_argument("--linear_prob_batch_size", type=int, default=64)
    p.add_argument("--linear_prob_eval_batch_size", type=int, default=None)
    p.add_argument("--linear_prob_lr", type=float, default=1e-3)
    p.add_argument("--linear_prob_weight_decay", type=float, default=0.0)
    p.add_argument("--linear_prob_schedule", type=str, default="cosine",
                    choices=["none", "cosine"])
    p.add_argument("--linear_prob_warmup_steps", type=int, default=0)
    p.add_argument("--linear_prob_min_lr_ratio", type=float, default=0.0)
    p.add_argument("--pool_method", type=str, default="pool", choices=["cls", "pool"])

    # ── Linear CKA (teacher↔student) ──
    p.add_argument("--cka_num_samples", type=int, default=2048,
                    help="Number of train images for teacher↔student Linear CKA "
                         "at each eval step. 0 disables CKA measurement.")
    p.add_argument("--cka_batch_size", type=int, default=None,
                    help="Batch size for CKA measurement. Defaults to "
                         "--linear_prob_batch_size.")

    # ── Abort / warnings ──
    p.add_argument("--abort_on_nan", dest="abort_on_nan", action="store_true")
    p.add_argument("--no_abort_on_nan", dest="abort_on_nan", action="store_false")
    p.add_argument("--suppress_warnings", dest="suppress_warnings", action="store_true")
    p.add_argument("--show_warnings", dest="suppress_warnings", action="store_false")

    # ── torch.compile ──
    p.add_argument("--compile", dest="compile", action="store_true",
                    help="Apply torch.compile to teacher, student, and projector.")
    p.add_argument("--no_compile", dest="compile", action="store_false",
                    help="Disable torch.compile (default).")
    p.add_argument("--compile_mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode.")

    # ── DDP ──
    p.add_argument("--ddp_find_unused_parameters", dest="ddp_find_unused_parameters",
                    action="store_true")
    p.add_argument("--no_ddp_find_unused_parameters", dest="ddp_find_unused_parameters",
                    action="store_false")

    # ── Weights & Biases ──
    p.add_argument("--wandb", dest="wandb", action="store_true",
                    help="Enable Weights & Biases logging.")
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                    help="Disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="speculative-encoding-distill")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)

    p.set_defaults(
        abort_on_nan=True,
        suppress_warnings=True,
        compile=False,
        ddp_find_unused_parameters=False,
        save_projector=True,
        wandb=True,
        refresh_image_cache=False,
        muon_nesterov=True,
        new_loss=False,
        only_affinity=False,
        only_distance=False,
    )
    args = p.parse_args()
    if args.only_affinity and args.only_distance:
        p.error("--only_affinity and --only_distance are mutually exclusive.")

    # ── Derive run/output tags from the full training signature ──
    config_tag = _build_config_tag(args)
    run_tag = _build_run_tag(args, config_tag)
    args.output_dir = os.path.join(args.output_base, run_tag)
    args.image_cache_dir = os.path.join(args.output_base, "_image_cache")
    if not args.wandb_run_name:
        args.wandb_run_name = run_tag

    return args
