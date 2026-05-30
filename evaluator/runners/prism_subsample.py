"""Evaluate PRISM (paige-ai/Prism) on CM16 / CM17 / TCGA-NSCLC at any patch budget.

Mirrors the structure of ``titan_subsample.py`` and ``gigapath_subsample.py``
so the unified ``eval.py`` CLI can dispatch all three slide-encoder rows of
Tab. 1 the same way.

Protocol
--------
1. Per-slide Virchow tile features ``[N, 2560]`` are loaded from
   ``$PRISM_FEATURE_ROOT`` (or ``--feature_root``).
2. Optionally subsampled per ``--sampling_mode`` (``custom`` reads pre-computed
   indices from ``--custom_index_root``; ``random`` is uniform).
3. Fed into ``model.prism.PRISM`` to get a slide embedding via
   ``slide_representations``.
4. A ``LogisticRegression`` linear probe is fit on each train fold (full bag)
   and evaluated on the validation fold at the requested budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# speculative_encoding/ — load_paths.sh prepends this to PYTHONPATH already; we
# repeat it so the script also runs as `python evaluator/runners/prism_subsample.py …`.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.metrics import get_eval_metrics_from_probs
from evaluator.runners.custom_index_utils import build_custom_index_cache
from evaluator.runners.mil_comparison import (
    CM16_RAW_ROOT,
    collect_cm16,
    collect_cm17,
    collect_nsclc,
    get_or_create_shared_splits,
    load_slide,
)
from model.prism import PRISM


# ── Default feature root resolution ─────────────────────────────────────────
def _resolve_default_prism_feature_root() -> str:
    """Pick the Virchow / PRISM-compatible feature root from env vars.

    Honoured in order: ``$PRISM_FEATURE_ROOT`` → ``$VIRCHOW_FEATURE_ROOT`` →
    ``$CLAM_ROOT/_multigpu/patch_features/ps224/virchow`` (legacy default).
    Falls back to ``""`` so ``--feature_root`` becomes effectively required.
    """
    explicit = os.environ.get("PRISM_FEATURE_ROOT") or os.environ.get("VIRCHOW_FEATURE_ROOT")
    if explicit:
        return explicit
    clam = os.environ.get("CLAM_ROOT", "")
    if clam:
        return f"{clam}_multigpu/patch_features/ps224/virchow"
    return ""


DEFAULT_PRISM_FEATURE_ROOT = _resolve_default_prism_feature_root()


# ── Metric helpers ──────────────────────────────────────────────────────────

def filter_scalar_metrics(metrics: Dict) -> Dict[str, float]:
    return {
        k: float(v) for k, v in metrics.items()
        if isinstance(v, (float, int, np.floating, np.integer))
    }


def aggregate_runs(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        return {"mean": {}, "std": {}}
    keys = [key for key in runs[0] if isinstance(runs[0][key], float)]
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in keys:
        vals = [run[key] for run in runs if key in run and not math.isnan(run[key])]
        if vals:
            mean[key] = float(np.mean(vals))
            std[key] = float(np.std(vals))
    return {"mean": mean, "std": std}


def print_aggregate(label: str, agg: Dict[str, Dict[str, float]]) -> None:
    mean = agg.get("mean", {})
    std = agg.get("std", {})
    keys = ["acc", "precision", "recall", "macro_f1", "auroc",
            "avg_patches", "latency", "latency_per_slide"]
    parts = [f"{k}={mean[k]:.4f}±{std.get(k, 0.0):.4f}" for k in keys if k in mean]
    print(f"    {label}: " + "  ".join(parts))


# ── Subsampling (PRISM-specific subset) ─────────────────────────────────────

def _normalize_sampling_mode(mode: str) -> str:
    mode = (mode or "").replace("-", "_")
    return "geometric" if mode == "grid" else mode


def _select_indices(
    n_features: int,
    patch_ratio: float,
    rng: np.random.Generator,
    sampling_mode: str,
    custom_indices: Optional[np.ndarray],
) -> np.ndarray:
    """Pick the patch indices to keep for one slide.

    PRISM's encoder consumes pre-extracted Virchow tiles, so we only support
    ``custom`` (pre-computed indices from the speculative-encoding sampler)
    and ``random`` (uniform). ``grid`` / ``k_means`` / ``k_medoid`` are
    available in the TITAN/Gigapath runners — for PRISM, sample first with
    ``python -m sampling`` and pass ``--custom_index_root``.
    """
    mode = _normalize_sampling_mode(sampling_mode)
    if patch_ratio >= 1.0 and mode != "custom":
        return np.arange(n_features, dtype=np.int64)
    if mode == "custom":
        if custom_indices is None or custom_indices.size == 0:
            raise RuntimeError(
                "sampling_mode=custom requires a populated --custom_index_root, "
                "but the slide had no matching index file."
            )
        idx = custom_indices.astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_features)]
        return np.unique(idx)
    if mode == "random":
        k = max(1, int(round(n_features * patch_ratio)))
        return np.sort(rng.choice(n_features, k, replace=False).astype(np.int64))
    raise NotImplementedError(
        f"sampling_mode={sampling_mode!r} is not supported in PRISM eval. "
        "Use 'custom' (with --custom_index_root) or 'random'."
    )


# ── PRISM forward pass ──────────────────────────────────────────────────────

def prism_extract_embeddings(
    prism: PRISM,
    paths: List[Path],
    labels: List[int],
    *,
    patch_ratio: float = 1.0,
    sampling_mode: str = "custom",
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Run PRISM on a list of slides at the requested patch budget.

    Returns ``(embeddings [M, 1280], labels [M], n_patches [M])``.
    """
    embs: List[np.ndarray] = []
    ys: List[int] = []
    n_patches: List[int] = []
    rng = np.random.default_rng(seed)
    cache = custom_index_cache or {}

    for path, label in zip(paths, labels):
        feat = load_slide(path)
        if feat.ndim != 2 or feat.shape[0] == 0:
            continue
        n_orig = int(feat.shape[0])
        if patch_ratio < 1.0 or sampling_mode == "custom":
            keep = _select_indices(
                n_features=n_orig,
                patch_ratio=patch_ratio,
                rng=rng,
                sampling_mode=sampling_mode,
                custom_indices=cache.get(path.stem),
            )
            feat = feat[torch.from_numpy(keep).long()]

        with torch.inference_mode():
            slide_emb = prism.forward_slide(feat)  # (1, 1280)
        embs.append(slide_emb.squeeze(0).float().cpu().numpy())
        ys.append(int(label))
        n_patches.append(int(feat.shape[0]))

    if not embs:
        raise RuntimeError("PRISM embedding extraction produced 0 slides.")
    return np.stack(embs).astype(np.float32), np.array(ys, dtype=np.int64), n_patches


def infer_feature_dim(paths: List[Path]) -> int:
    for path in paths:
        feat = load_slide(path)
        if feat.ndim == 2 and feat.shape[0] > 0:
            return int(feat.shape[1])
    return -1


def load_or_extract_prism_embeddings(
    cache_path: Path,
    prism: PRISM,
    paths: List[Path],
    labels: List[int],
    *,
    patch_ratio: float = 1.0,
    sampling_mode: str = "custom",
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Cache slide embeddings for one fold.

    ``cache_path`` should encode ``(split, fold, budget, sampler)`` so that
    re-running at a different budget does not silently reuse stale arrays.
    """
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=True)
        embs = saved["embeddings"]
        ys = saved["labels"]
        n_patches = saved["n_patches"].tolist()
        saved.close()
        return embs, ys, n_patches

    embs, ys, n_patches = prism_extract_embeddings(
        prism=prism, paths=paths, labels=labels,
        patch_ratio=patch_ratio,
        sampling_mode=sampling_mode,
        custom_index_cache=custom_index_cache,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embs,
        labels=ys,
        n_patches=np.array(n_patches, dtype=np.int32),
    )
    return embs, ys, n_patches


# ── Linear probe ────────────────────────────────────────────────────────────

def load_or_fit_prism_probe(
    probe_path: Path,
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    probe_max_iter: int,
):
    if probe_path.exists():
        with open(probe_path, "rb") as f:
            return pickle.load(f)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train_embs)
    probe = LogisticRegression(
        max_iter=probe_max_iter, C=1.0, random_state=seed, class_weight="balanced",
    )
    probe.fit(x_train_scaled, train_labels)
    saved = {"scaler": scaler, "probe": probe, "seed": int(seed)}
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "wb") as f:
        pickle.dump(saved, f)
    return saved


def evaluate_fold_once(
    saved_probe,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
    val_n_patches: List[int],
) -> Dict[str, float]:
    t0 = time.perf_counter()
    scaler = saved_probe["scaler"]
    probe = saved_probe["probe"]
    x_val_scaled = scaler.transform(val_embs)
    probs = probe.predict_proba(x_val_scaled)
    metrics = get_eval_metrics_from_probs(probs, val_labels, n_class=probs.shape[1], prefix="")
    out = filter_scalar_metrics(metrics)
    elapsed = time.perf_counter() - t0
    out["avg_patches"] = float(np.mean(val_n_patches)) if val_n_patches else float("nan")
    out["latency"] = float(elapsed)
    out["latency_per_slide"] = float(elapsed / len(val_labels)) if len(val_labels) > 0 else float("nan")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PRISM (paige-ai/Prism) on CM16/CM17/NSCLC at any patch budget.",
    )
    # Dataset + features.
    parser.add_argument("--dataset", default="nsclc", choices=["cm16", "cm17", "nsclc"])
    parser.add_argument("--feature_root", default=DEFAULT_PRISM_FEATURE_ROOT,
                        help="Virchow-tile (2560-d) feature root.")
    parser.add_argument("--cm16_raw_root", default=CM16_RAW_ROOT,
                        help="CM16 lesion-annotation zip root (only when --dataset cm16).")
    parser.add_argument("--label_csv", default=None,
                        help="CM17 stages.csv (only when --dataset cm17).")
    # Splits / seeds.
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_root", default=None,
                        help="Directory to store / load shared splits. "
                             "Default: ./splits (SHARED_SPLIT_ROOT in mil_comparison.py).")
    parser.add_argument("--n_eval_seeds", type=int, default=1)
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    # Subsampling (for the +Ours rows of Tab. 1).
    parser.add_argument("--patch_ratio", type=float, default=1.0,
                        help="Patch budget. 1.0 = full bag (baseline row).")
    parser.add_argument("--sampling_mode", default="custom",
                        choices=["custom", "random"],
                        help="'custom' uses --custom_index_root; 'random' = uniform.")
    parser.add_argument("--custom_index_root", default=None,
                        help="Pre-computed index .npy directory written by `python -m sampling`.")
    # Model loading.
    parser.add_argument("--prism_model_dir", default=None,
                        help="Optional local PRISM checkpoint dir. "
                             "Default: HF Hub via $PRISM_HF_REPO (paige-ai/Prism).")
    # IO.
    parser.add_argument("--output_dir", default="./results_for_paper/prism")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not args.feature_root or not Path(args.feature_root).exists():
        raise RuntimeError(
            f"feature_root does not exist: {args.feature_root!r}. "
            "Pass a valid PRISM/Virchow feature root with --feature_root or "
            "set $PRISM_FEATURE_ROOT in configs/paths.json."
        )

    # Collect (paths, labels, names, class_names) for the requested dataset.
    if args.dataset == "nsclc":
        paths, labels, names, class_names = collect_nsclc(args.feature_root)
    elif args.dataset == "cm16":
        paths, labels, names, class_names = collect_cm16(
            args.feature_root, split="all", raw_root=args.cm16_raw_root,
        )
    elif args.dataset == "cm17":
        paths, labels, names, class_names = collect_cm17(args.feature_root, args.label_csv)
    else:
        raise SystemExit(f"unsupported dataset: {args.dataset}")

    feat_dim = infer_feature_dim(paths)
    if feat_dim != PRISM.TILE_EMBEDDING_DIM:
        raise RuntimeError(
            f"PRISM expects {PRISM.TILE_EMBEDDING_DIM}-d Virchow tile embeddings, "
            f"but got feature_dim={feat_dim} from --feature_root ({args.feature_root}). "
            "Use Virchow/PRISM-compatible features (224-px tiles)."
        )

    # Custom-index cache (only when sampling_mode=custom and patch_ratio<1.0).
    custom_index_cache = None
    if args.sampling_mode == "custom" and args.custom_index_root and args.patch_ratio < 1.0:
        custom_index_cache = build_custom_index_cache(
            paths,
            feature_root=Path(args.feature_root),
            custom_index_root=Path(args.custom_index_root),
        )
        if not custom_index_cache:
            raise RuntimeError(
                "sampling_mode=custom requires matching index files under "
                f"--custom_index_root={args.custom_index_root}, but none were found."
            )

    folds, split_desc, split_path = get_or_create_shared_splits(
        dataset=args.dataset,
        sample_names=names,
        labels=labels,
        n_splits=args.n_folds,
        test_size=args.test_size,
        seed=args.seed,
        split_root=args.split_root,
        sample_paths=paths,
        class_names=class_names,
    )

    print(f"\n{'=' * 68}")
    print("  PRISM eval")
    print(f"  Dataset       : {args.dataset.upper()}")
    print(f"  Feature root  : {args.feature_root}")
    print(f"  Split         : {split_desc}")
    print(f"  Split file    : {split_path}")
    print(f"  Eval seeds    : {args.n_eval_seeds}")
    print(f"  Patch ratio   : {args.patch_ratio}")
    print(f"  Sampling mode : {args.sampling_mode}")
    print(f"  Device        : {device}")
    print(f"{'=' * 68}")

    prism = PRISM(
        model_dir=Path(args.prism_model_dir) if args.prism_model_dir else None,
        device=device,
    )

    fold_details: List[Dict] = []
    fold_means: List[Dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        tr_paths = [paths[i] for i in train_idx]
        tr_lbl = [labels[i] for i in train_idx]
        va_paths = [paths[i] for i in val_idx]
        va_lbl = [labels[i] for i in val_idx]
        print(f"\n  fold={fold_idx}  train={len(tr_paths)}  val={len(va_paths)}")

        # Cache key encodes (split, fold, budget, sampler) so that switching
        # --patch_ratio doesn't reuse stale arrays. The train fold is always
        # cached at full bag — the linear probe is fit on the un-subsampled
        # distribution and reused at every test budget.
        budget_tag = "fullbag" if args.patch_ratio >= 1.0 else f"b{args.patch_ratio:g}_{args.sampling_mode}"
        train_cache = cache_dir / f"fold{fold_idx}_train_fullbag.npz"
        val_cache = cache_dir / f"fold{fold_idx}_val_{budget_tag}.npz"

        # Training fold: always full bag, no sampling (linear probe is fit on
        # the un-subsampled distribution). `random` + 1.0 is a no-op pass-through
        # in _select_indices, unlike `custom` which would require indices.
        train_embs, train_lbl_np, _ = load_or_extract_prism_embeddings(
            cache_path=train_cache,
            prism=prism,
            paths=tr_paths,
            labels=tr_lbl,
            patch_ratio=1.0,
            sampling_mode="random",
            custom_index_cache=None,
            seed=args.seed,
        )
        val_embs, val_lbl_np, val_n_patches = load_or_extract_prism_embeddings(
            cache_path=val_cache,
            prism=prism,
            paths=va_paths,
            labels=va_lbl,
            patch_ratio=args.patch_ratio,
            sampling_mode=args.sampling_mode,
            custom_index_cache=custom_index_cache,
            seed=args.seed,
        )

        seed_runs: List[Dict] = []
        probe_cache_paths: List[str] = []
        for eval_seed in range(args.seed, args.seed + args.n_eval_seeds):
            probe_cache = cache_dir / f"fold{fold_idx}_probe_seed{eval_seed}.pkl"
            saved_probe = load_or_fit_prism_probe(
                probe_path=probe_cache,
                train_embs=train_embs,
                train_labels=train_lbl_np,
                seed=eval_seed,
                probe_max_iter=args.probe_max_iter,
            )
            run = evaluate_fold_once(
                saved_probe=saved_probe,
                val_embs=val_embs,
                val_labels=val_lbl_np,
                val_n_patches=val_n_patches,
            )
            seed_runs.append(run)
            probe_cache_paths.append(str(probe_cache))
            print(
                f"    seed={eval_seed}  acc={run.get('acc', float('nan')):.4f}  "
                f"macro_f1={run.get('macro_f1', float('nan')):.4f}  "
                f"auroc={run.get('auroc', float('nan')):.4f}  "
                f"avg_patches={run.get('avg_patches', float('nan')):.1f}  "
                f"latency={run.get('latency', float('nan')):.2f}s"
            )

        fold_agg = aggregate_runs(seed_runs)
        fold_means.append(fold_agg["mean"])
        fold_details.append({
            "fold": fold_idx,
            "n_train": len(tr_paths),
            "n_val": len(va_paths),
            "probe_cache_paths": probe_cache_paths,
            "seed_runs": seed_runs,
            "aggregate": fold_agg,
        })
        print_aggregate(f"fold {fold_idx}", fold_agg)

    overall = aggregate_runs(fold_means)
    print_aggregate("overall", overall)

    summary = {
        "dataset": args.dataset,
        "feature_root": args.feature_root,
        "n_folds": args.n_folds,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_eval_seeds": args.n_eval_seeds,
        "probe_max_iter": args.probe_max_iter,
        "patch_ratio": args.patch_ratio,
        "sampling_mode": args.sampling_mode,
        "split": split_desc,
        "split_file": str(split_path),
        "split_dir": str(split_path.with_suffix("")),
        "class_names": class_names,
        "results": {
            "prism": {
                "folds": fold_details,
                "aggregate": overall,
            },
        },
    }
    summary_path = out_dir / "subsample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved -> {summary_path}")


if __name__ == "__main__":
    main()
