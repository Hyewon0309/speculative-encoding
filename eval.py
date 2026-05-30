#!/usr/bin/env python3
"""Single CLI entry point for every Speculative-Encoding experiment in the paper.

Usage
-----

    python eval.py \
        --dataset cm16 \
        --model titan \
        --budget 0.25 \
        --sampling-mode custom \
        --sampling-config configs/sampling/canonical_25pct.json

Or load every flag from a per-experiment JSON:

    python eval.py --config configs/experiments/cm16_titan_ours_b25.json

Or override any value loaded from the config:

    python eval.py --config configs/experiments/cm16_titan_ours_b25.json \
                   --output-dir outputs/my_run --n-eval-seeds 3

What the CLI does
-----------------

1. Loads ``configs/paths.json`` if available (so $FEATURE_ROOT / $COORD_ROOT /
   $CHECKPOINT_DIR / etc. propagate to every subcommand consistently). You can
   also point ``PATHS_JSON`` at a different file.

2. If ``--sampling-mode custom`` is requested and no ``--custom-index-root`` is
   provided, runs ``python -m sampling`` first to materialise the index files
   for the requested ``--budget``.

3. Dispatches to the correct evaluation runner based on ``--model``:

       abmil / clam_sb / dftd / dsmil / ilra / rrt / mha / transmil / wikg
                            -> evaluator/runners/mil_subsample.py     (needs --checkpoint-dir)
       titan                -> evaluator/runners/titan_subsample.py
       prism                -> evaluator/runners/prism_subsample.py
       gigapath             -> evaluator/runners/gigapath_subsample.py
       all                  -> 9 MIL aggregators + titan + prism + gigapath

4. Writes one JSON summary per (dataset, model, budget) cell under
   ``--output-dir``.

Models
------
  • The 9 MIL aggregators are the c25A baseline ones from Tab. 1 of the paper.
  • ``mha`` is the plain Transformer aggregator, kept under its legacy code
    name for backward compatibility with the runners.

Hyperparameters
---------------
The ``configs/sampling/`` folder pins the sampler hyperparameters used for
each row of the paper. ``configs/experiments/main_table/*.json`` bundles
those with the dataset / model / budget triple for one-line reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent

# 9 MIL aggregators trained from scratch.
MIL_ARCHS: List[str] = [
    "abmil", "clam_sb", "dftd", "dsmil", "ilra", "rrt", "mha", "transmil", "wikg",
]
# Slide encoders (linear-probed at the sampled budget).
SLIDE_ENCODERS: List[str] = ["titan", "prism", "gigapath"]
ALL_MODELS: List[str] = MIL_ARCHS + SLIDE_ENCODERS

# Datasets we publicly support.
DATASETS = ("cm16", "cm17", "nsclc")


# ─────────────────────────────────────────────────────────────────────────────
# configs/paths.json -> os.environ
# ─────────────────────────────────────────────────────────────────────────────

_VAR_PAT = __import__("re").compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _load_paths_json(path: Path) -> None:
    """Export every leaf in ``configs/paths.json`` as an env var.

    Mirrors ``scripts/load_paths.sh`` but in Python so the CLI works on
    machines without the bash helper sourced. Insertion order is preserved
    so ``${OTHER_KEY}`` substitution resolves correctly.
    """
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for key, val in data.items():
        if key.startswith("_") or not isinstance(val, str):
            continue
        resolved = _VAR_PAT.sub(lambda m: os.environ.get(m.group(1), m.group(0)), val)
        # Don't clobber values the user has already exported in their shell.
        os.environ.setdefault(key, resolved)
        if key == "PYTHONPATH_PREPEND":
            cur = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = resolved if not cur else f"{resolved}:{cur}"


def _autoload_paths() -> Path:
    """Pick the paths.json to use.

    Resolution order:
      1. ``$PATHS_JSON``                     (explicit override)
      2. ``configs/paths.local.json``        (per-machine local config; .gitignored)
      3. ``configs/paths.json``              (committed template)
    """
    explicit = os.environ.get("PATHS_JSON")
    if explicit:
        candidate = Path(explicit)
    else:
        local = REPO_ROOT / "configs" / "paths.local.json"
        candidate = local if local.exists() else REPO_ROOT / "configs" / "paths.json"
    _load_paths_json(candidate)
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speculative-encoding",
        description="Evaluate Speculative-Encoding on CM16 / CM17 / TCGA-NSCLC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Top-level ────────────────────────────────────────────────────────────
    p.add_argument(
        "--config",
        type=str, default=None,
        help="Per-experiment JSON (e.g. configs/experiments/cm16_titan_ours_b25.json). "
             "Every flag below can be set in the JSON; explicit CLI flags take "
             "precedence over the config.",
    )

    # ── Dataset / model / budget (the three knobs you change most) ───────────
    p.add_argument("--dataset",  choices=DATASETS, default=None,
                   help="Dataset to evaluate on.")
    p.add_argument("--model",
                   choices=ALL_MODELS + ["all", "mil_all", "encoder_all"],
                   default=None,
                   help="Which model to evaluate. 'all' = all 12, "
                        "'mil_all' = the 9 MIL aggregators, "
                        "'encoder_all' = titan/prism/gigapath.")
    p.add_argument("--budget", type=float, default=None,
                   help="Patch budget (e.g. 0.25 for 25%%).")

    # ── Sampler ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--sampling-mode", default="custom",
        choices=["custom", "random", "grid", "k_means", "k_medoid"],
        help="'custom' = use Speculative-Encoding indices (default); "
             "'random' / 'grid' / 'k_means' / 'k_medoid' = paper baselines.",
    )
    p.add_argument(
        "--sampling-config", type=str, default=None,
        help="Path to a sampler JSON under configs/sampling/* or configs/ablation/*. "
             "Required when --sampling-mode=custom and no --custom-index-root.",
    )
    p.add_argument(
        "--custom-index-root", type=str, default=None,
        help="Pre-computed index .npy directory (skips re-running the sampler).",
    )

    # ── Data roots (env vars used as defaults) ───────────────────────────────
    p.add_argument(
        "--feature-root", default=None,
        help="Patch-feature root. Default: $FEATURE_ROOT.",
    )
    p.add_argument(
        "--coord-root", default=None,
        help="Patch-coord root. Default: $COORD_ROOT.",
    )
    p.add_argument(
        "--gigapath-feature-root", default=None,
        help="Prov-GigaPath patch features (256 px). Default: $GIGAPATH_FEATURE_ROOT.",
    )
    p.add_argument(
        "--prism-feature-root", default=None,
        help="PRISM/Virchow patch features (224 px). Default: $PRISM_FEATURE_ROOT.",
    )
    p.add_argument(
        "--cm16-raw-root", default=None,
        help="CAMELYON16 raw lesion-annotation zip root. Default: $CM16_RAW_ROOT.",
    )
    p.add_argument(
        "--distilled-feature-root", default=None,
        help="Distilled student-encoder features used by the sampler. Default: $DISTILLED_FEATURE_ROOT.",
    )

    # ── MIL aggregator specifics ─────────────────────────────────────────────
    p.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory with <arch>_fold<N>_best.pt files (from `train_mil_checkpoints.sh`). "
             "Required for any MIL --model. Default: $CHECKPOINT_DIR.",
    )
    p.add_argument("--checkpoint-folds", type=int, nargs="+", default=None,
                   help="Folds to average over (default: 0..4 = official 5-fold mean).")

    # ── Eval knobs ───────────────────────────────────────────────────────────
    p.add_argument("--n-eval-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu.")
    p.add_argument("--gpu-id", type=int, default=0,
                   help="Used only when sampling has to run.")

    # ── Output ──────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default=None,
                   help="Where to write JSON summaries. Default: outputs/<dataset>/<model>/<budget>.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true",
                   help="Print the subprocess commands but don't execute them.")

    return p


def _merge_config(args: argparse.Namespace, raw_argv: List[str]) -> argparse.Namespace:
    """When --config is given, fold its keys into ``args`` so explicit CLI
    flags still win.
    """
    if args.config is None:
        return args
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    # Translate JSON keys (snake_case or kebab-case) to argparse attr names.
    given = set()
    for tok in raw_argv:
        if tok.startswith("--"):
            given.add(tok[2:].replace("-", "_").split("=")[0])

    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        attr = k.replace("-", "_")
        if attr in given:
            continue  # CLI overrides config.
        if not hasattr(args, attr):
            print(f"[eval] [warn] config key not recognised, ignoring: {k}", file=sys.stderr)
            continue
        setattr(args, attr, v)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def _python() -> str:
    """Resolve the interpreter for every subprocess (single-env install)."""
    return os.environ.get("PYTHON", sys.executable)


def _ensure(name: str, value: Optional[str], hint: str) -> str:
    if value:
        return value
    raise SystemExit(
        f"[eval] {name} is not set. {hint} "
        f"(pass it on the command line or export it via configs/paths.json)."
    )


def _run(cmd: List[str], dry_run: bool, env: Optional[Dict[str, str]] = None) -> None:
    pretty = " ".join(shlex.quote(t) for t in cmd)
    print(f"[eval] $ {pretty}")
    if dry_run:
        return
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"[eval] subcommand failed (exit {rc}): {pretty}")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling step
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_run_sampler(args: argparse.Namespace) -> str:
    """Run ``python -m sampling`` over the distilled features if we need to
    materialise custom indices, and return the path containing ``.npy`` files."""
    if args.custom_index_root:
        return args.custom_index_root

    if args.sampling_mode != "custom":
        # Random / grid / k-means / k-medoid run inside the eval runners
        # themselves at __getitem__ time; no separate sampler invocation.
        return ""

    # Need to sample. Resolve config.
    if not args.sampling_config:
        raise SystemExit(
            "[eval] --sampling-config is required when --sampling-mode=custom "
            "and no --custom-index-root is given."
        )
    cfg_path = Path(args.sampling_config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"[eval] sampling config not found: {cfg_path}")

    distilled_root = args.distilled_feature_root or os.environ.get("DISTILLED_FEATURE_ROOT", "")
    distilled_root = _ensure(
        "DISTILLED_FEATURE_ROOT", distilled_root,
        "Where the distilled-student patch features live (input to the sampler).",
    )
    # Expected layout: <distilled_root>/<dataset>/  (cm16 / cm17 / NSCLC).
    if args.dataset == "nsclc":
        ds_dir = Path(distilled_root) / "NSCLC"
    else:
        ds_dir = Path(distilled_root) / args.dataset

    out_root = Path(args.output_dir or REPO_ROOT / "outputs" / args.dataset / "indices")
    out_dir = out_root / f"{args.budget:.2f}"

    cfg = json.loads(cfg_path.read_text())
    flags: List[str] = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                flags.append(flag)
        else:
            flags.extend([flag, str(v)])

    cmd = [
        _python(), "-m", "sampling",
        "--input_dir", str(ds_dir),
        "--output_dir", str(out_dir),
        "--device", f"cuda:{args.gpu_id}",
        "--recursive", "--overwrite",
        *flags,
    ]
    _run(cmd, args.dry_run)
    return str(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_mil(model: str, args: argparse.Namespace, index_root: str, out_dir: Path) -> None:
    archs = MIL_ARCHS if model in {"mil_all", "all"} else [model]
    feature_root = args.feature_root or os.environ.get("FEATURE_ROOT", "")
    feature_root = _ensure("FEATURE_ROOT", feature_root, "CONCH v1.5 patch-feature directory.")
    ckpt_dir = args.checkpoint_dir or os.environ.get("CHECKPOINT_DIR", "")
    ckpt_dir = _ensure(
        "CHECKPOINT_DIR", ckpt_dir,
        "Pre-trained MIL checkpoint dir; train them first with "
        "`bash scripts/train_mil_checkpoints.sh`.",
    )
    folds = args.checkpoint_folds or [0, 1, 2, 3, 4]

    cmd = [
        _python(), str(REPO_ROOT / "evaluator" / "runners" / "mil_subsample.py"),
        "--dataset", args.dataset,
        "--mil_archs", *archs,
        "--patch_ratio", str(args.budget),
        "--sampling_mode", args.sampling_mode,
        "--feature_root", feature_root,
        "--n_eval_seeds", str(args.n_eval_seeds),
        "--checkpoint_dir", ckpt_dir,
        "--checkpoint_folds", *(str(f) for f in folds),
        "--output_dir", str(out_dir),
    ]
    if args.coord_root or os.environ.get("COORD_ROOT"):
        cmd += ["--coord_root", args.coord_root or os.environ["COORD_ROOT"]]
    if index_root:
        cmd += ["--custom_index_root", index_root]
    if args.cm16_raw_root or os.environ.get("CM16_RAW_ROOT"):
        cmd += ["--cm16_raw_root", args.cm16_raw_root or os.environ["CM16_RAW_ROOT"]]
    if args.dataset == "cm16":
        cmd += ["--cm16_split_mode", "official"]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.dry_run)


def _dispatch_titan(args: argparse.Namespace, index_root: str, out_dir: Path) -> None:
    feature_root = args.feature_root or os.environ.get("FEATURE_ROOT", "")
    feature_root = _ensure("FEATURE_ROOT", feature_root, "CONCH v1.5 patch features for TITAN.")
    cmd = [
        _python(),
        str(REPO_ROOT / "evaluator" / "runners" / "titan_subsample.py"),
        "--dataset", args.dataset,
        "--patch_ratio", str(args.budget),
        "--sampling_mode", args.sampling_mode,
        "--feature_root", feature_root,
        "--output_dir", str(out_dir),
    ]
    if index_root:
        cmd += ["--custom_index_root", index_root]
    if args.coord_root or os.environ.get("COORD_ROOT"):
        cmd += ["--coord_root", args.coord_root or os.environ["COORD_ROOT"]]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.dry_run)


def _dispatch_gigapath(args: argparse.Namespace, index_root: str, out_dir: Path) -> None:
    feature_root = args.gigapath_feature_root or os.environ.get("GIGAPATH_FEATURE_ROOT", "")
    feature_root = _ensure(
        "GIGAPATH_FEATURE_ROOT", feature_root,
        "Prov-GigaPath patch features (256 px / ViT-giant).",
    )
    cmd = [
        _python(),
        str(REPO_ROOT / "evaluator" / "runners" / "gigapath_subsample.py"),
        "--dataset", args.dataset,
        "--patch_ratio", str(args.budget),
        "--sampling_mode", args.sampling_mode,
        "--feature_root", feature_root,
        "--output_dir", str(out_dir),
    ]
    if index_root:
        cmd += ["--custom_index_root", index_root]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.dry_run)


def _dispatch_prism(args: argparse.Namespace, index_root: str, out_dir: Path) -> None:
    feature_root = (
        args.prism_feature_root
        or os.environ.get("PRISM_FEATURE_ROOT")
        or os.environ.get("VIRCHOW_FEATURE_ROOT", "")
    )
    feature_root = _ensure(
        "PRISM_FEATURE_ROOT", feature_root,
        "Virchow tile features (2560-d) consumed by paige-ai/Prism.",
    )
    cmd = [
        _python(),
        str(REPO_ROOT / "evaluator" / "runners" / "prism_subsample.py"),
        "--dataset", args.dataset,
        "--feature_root", feature_root,
        "--patch_ratio", str(args.budget),
        "--sampling_mode", args.sampling_mode,
        "--n_eval_seeds", str(args.n_eval_seeds),
        "--output_dir", str(out_dir),
    ]
    if index_root:
        cmd += ["--custom_index_root", index_root]
    if args.cm16_raw_root or os.environ.get("CM16_RAW_ROOT"):
        cmd += ["--cm16_raw_root", args.cm16_raw_root or os.environ["CM16_RAW_ROOT"]]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    _autoload_paths()
    args = _merge_config(args, raw_argv)

    # Validate the three required knobs after config merge.
    for name in ("dataset", "model", "budget"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required (or set it in --config).")

    # Resolve output dir.
    budget_tag = f"{args.budget:.2f}"
    if not args.output_dir:
        args.output_dir = str(REPO_ROOT / "outputs" / args.dataset / args.model / budget_tag)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1 — sampling (only when needed).
    index_root = _maybe_run_sampler(args)

    # Step 2 — model-specific eval(s).
    if args.model == "all":
        models = ALL_MODELS
    elif args.model == "mil_all":
        models = MIL_ARCHS
    elif args.model == "encoder_all":
        models = SLIDE_ENCODERS
    else:
        models = [args.model]

    for m in models:
        sub_out = out_root if len(models) == 1 else out_root / m
        sub_out.mkdir(parents=True, exist_ok=True)
        if m in MIL_ARCHS:
            _dispatch_mil(m, args, index_root, sub_out)
        elif m == "titan":
            _dispatch_titan(args, index_root, sub_out)
        elif m == "gigapath":
            _dispatch_gigapath(args, index_root, sub_out)
        elif m == "prism":
            _dispatch_prism(args, index_root, sub_out)
        else:
            raise SystemExit(f"[eval] unknown model: {m}")

    print(f"[eval] done -> {out_root}")


if __name__ == "__main__":
    main()
