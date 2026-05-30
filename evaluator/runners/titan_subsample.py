"""
Evaluate TITAN on CM16/CM17/PANDA/NSCLC with test-time patch subsampling.

Train-time setup:
  - TITAN slide embeddings are extracted with patch_ratio=1.0
  - A linear probe is fit on the train split / train fold

Test-time setup:
  - TITAN slide embeddings are extracted again using the requested patch_ratio
  - The saved linear probe is applied to those subsampled slide embeddings
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# speculative_encoding/ — load_paths.sh prepends this to PYTHONPATH already; we
# repeat it so the script also runs as `python evaluator/runners/titan_subsample.py …`.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MIL-Lab is not on the TITAN linear-probe path; the legacy sys.path injection
# was removed when evaluator/abmil_eval.py was retired. The env var is still
# honoured by callers that need it (none in this file).

from evaluator.metrics import get_eval_metrics_from_probs
from evaluator.runners.custom_index_utils import build_custom_index_cache
from evaluator.runners.feasibility_subsample import find_coord_file, load_coords, titan_extract_embeddings
from evaluator.runners.mil_comparison import (
    CM16_RAW_ROOT,
    FEATURE_ROOT,
    collect_cm16,
    collect_cm17,
    collect_nsclc,
    collect_panda,
    get_or_create_shared_splits,
)


# Default falls back to the env var. If COORD_ROOT is unset and the user does
# not pass --coord_root, downstream code raises a clear error.
DEFAULT_COORD_ROOT = os.environ.get("COORD_ROOT", "")


def _cuda_sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def normalize_sampling_mode(mode: str) -> str:
    mode = mode.replace("-", "_")
    return "geometric" if mode == "grid" else mode


def _resolve_model_dir(model_dir: Optional[str]) -> Optional[Path]:
    if model_dir is None:
        return None
    if model_dir.strip().lower() in {"", "none", "null", "auto", "hf", "hub"}:
        return None
    return Path(model_dir)


def build_label_map(names: List[str], labels: List[int]) -> Dict[str, int]:
    return {name: label for name, label in zip(names, labels)}


def infer_coord_path(feature_path: Path, feature_root: Path, coord_root: Path) -> Optional[Path]:
    try:
        rel_path = feature_path.relative_to(feature_root)
        coord_path = (coord_root / rel_path).with_suffix(".npy")
        if coord_path.exists():
            return coord_path

        candidates = sorted(coord_path.parent.glob(f"{feature_path.stem}.*.npy"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple coord files match {feature_path.stem} under {coord_path.parent}: {candidates}"
            )
    except ValueError:
        pass

    fallback = find_coord_file(coord_root, feature_path.parent.name, feature_path.stem)
    if fallback is not None:
        return fallback
    return None


def build_coord_cache(
    paths: List[Path],
    feature_root: Path,
    coord_root: Path,
) -> Dict[str, Tuple[np.ndarray, int]]:
    coord_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    for path in paths:
        slide_id = path.stem
        if slide_id in coord_cache:
            continue
        coord_path = infer_coord_path(path, feature_root=feature_root, coord_root=coord_root)
        if coord_path is None:
            continue
        coord_cache[slide_id] = load_coords(coord_path)
    return coord_cache


def filter_scalar_metrics(metrics: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            out[key] = float(value)
    return out


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
    keys = ["acc", "macro_f1", "auroc", "avg_patches", "latency", "latency_per_slide"]
    parts = []
    for key in keys:
        if key in mean:
            parts.append(f"{key}={mean[key]:.4f}±{std.get(key, 0.0):.4f}")
    print(f"    {label}: " + "  ".join(parts))


def load_or_extract_train_embeddings(
    cache_path: Path,
    titan,
    train_paths: List[Path],
    label_map: Dict[str, int],
    coord_root: Path,
    patch_ratio: float,
    device: str,
    seed: int,
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    titan_slide_feat_dir: Optional[Path] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=True)
        embeddings = saved["embeddings"]
        labels = saved["labels"]
        slide_ids = saved["slide_ids"].tolist()
        saved.close()
        print(f"    loaded cached train embeddings -> {cache_path.name} {embeddings.shape}")
        return embeddings, labels, slide_ids

    embeddings, labels, _, slide_ids = titan_extract_embeddings(
        titan=titan,
        paths=train_paths,
        labels=label_map,
        coord_dir=coord_root,
        patch_ratio=patch_ratio,
        device=device,
        seed=seed,
        sampling_mode=normalize_sampling_mode("random"),
        coord_cache=coord_cache,
        slide_feat_dir=titan_slide_feat_dir,
        custom_index_cache=custom_index_cache,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        labels=labels,
        slide_ids=np.array(slide_ids),
    )
    print(f"    saved train embeddings -> {cache_path.name} {embeddings.shape}")
    return embeddings, labels, slide_ids


def load_or_fit_probe(
    probe_path: Path,
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    max_iter: int,
    require_cached_probe: bool = False,
):
    if probe_path.exists():
        print(f"    loaded cached probe -> {probe_path.name}")
        return joblib.load(probe_path)
    if require_cached_probe:
        raise FileNotFoundError(
            f"Required cached probe not found: {probe_path}. "
            "Set --require_cached_probe only when probe caches already exist."
        )

    scaler = StandardScaler()
    train_embs_scaled = scaler.fit_transform(train_embs)
    probe = LogisticRegression(
        max_iter=max_iter,
        C=1.0,
        random_state=seed,
        class_weight="balanced",
    )
    probe.fit(train_embs_scaled, train_labels)

    saved = {"probe": probe, "scaler": scaler}
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, probe_path)
    print(f"    fitted probe -> {probe_path.name}")
    return saved


def evaluate_with_probe(
    titan,
    saved_probe,
    eval_paths: List[Path],
    label_map: Dict[str, int],
    coord_root: Path,
    patch_ratio: float,
    device: str,
    seed: int,
    sampling_mode: str,
    fix_alibi: bool,
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    embs, labels, n_patch_list, _, gpu_seconds, selection_seconds = titan_extract_embeddings(
        titan=titan,
        paths=eval_paths,
        labels=label_map,
        coord_dir=coord_root,
        patch_ratio=patch_ratio,
        device=device,
        seed=seed,
        sampling_mode=normalize_sampling_mode(sampling_mode),
        fix_alibi=fix_alibi,
        coord_cache=coord_cache,
        custom_index_cache=custom_index_cache,
        measure_gpu_time=True,
    )
    probs = saved_probe["probe"].predict_proba(saved_probe["scaler"].transform(embs))
    metrics = get_eval_metrics_from_probs(
        probs,
        labels,
        n_class=probs.shape[1],
        get_report=False,
        prefix="",
    )
    out = filter_scalar_metrics(metrics)
    out["avg_patches"] = float(np.mean(n_patch_list)) if n_patch_list else float("nan")
    out["selection_time"] = float(selection_seconds)
    out["gpu_time"] = float(gpu_seconds)
    out["latency"] = float(gpu_seconds + selection_seconds)
    out["latency_per_slide"] = float((gpu_seconds + selection_seconds) / len(labels)) if len(labels) > 0 else float("nan")
    return out


def load_titan(model_dir: Optional[Path], device: str):
    from model.titan import TITAN

    if model_dir is None:
        print("[INFO] TITAN model_dir=None -> will use HF cache / HF Hub settings.")
        print(f"[INFO] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[INFO] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    return TITAN(model_dir=model_dir, device=device, pretrained=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TITAN with patch subsampling")
    parser.add_argument("--dataset", required=True, choices=["cm16", "cm17", "panda", "nsclc"])
    parser.add_argument("--feature_root", default=FEATURE_ROOT)
    parser.add_argument("--coord_root", default=DEFAULT_COORD_ROOT)
    parser.add_argument("--cm16_split_mode", default="cv", choices=["cv", "official"])
    parser.add_argument("--cm16_raw_root", default=CM16_RAW_ROOT)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--train_label_csv", default=None)
    parser.add_argument("--test_label_csv", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--patch_ratio", type=float, default=1.0)
    parser.add_argument("--train_patch_ratio", type=float, default=1.0)
    parser.add_argument("--n_eval_seeds", type=int, default=1)
    parser.add_argument(
        "--sampling_mode",
        default="random",
        choices=["random", "grid", "geometric", "k_means", "k_medoid", "k-medoid", "hdbscan", "custom"],
    )
    parser.add_argument("--fix_alibi", action="store_true")
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--titan_slide_feat_dir", default=None)
    parser.add_argument("--custom_index_root", default=None)
    parser.add_argument(
        "--require_cached_probe",
        action="store_true",
        help="If set, never fit a new probe. Fail when cached probe (.pkl) is missing.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = _resolve_model_dir(args.model_dir)
    feature_root = Path(args.feature_root)
    coord_root = Path(args.coord_root)
    titan_slide_feat_dir = Path(args.titan_slide_feat_dir) if args.titan_slide_feat_dir else None
    custom_index_root = Path(args.custom_index_root) if args.custom_index_root else None

    if args.output_dir is None:
        ratio_tag = f"{int(round(args.patch_ratio * 100)):02d}pct"
        args.output_dir = f"./results/{args.dataset}_titan_{ratio_tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    fixed_split = args.train_label_csv is not None and args.test_label_csv is not None
    cm16_official_split = args.dataset == "cm16" and args.cm16_split_mode == "official"
    if cm16_official_split:
        tr_paths, tr_labels, tr_names, class_names = collect_cm16(
            args.feature_root, split="train", raw_root=args.cm16_raw_root
        )
        te_paths, te_labels, te_names, _ = collect_cm16(
            args.feature_root, split="test", raw_root=args.cm16_raw_root
        )
        label_map = build_label_map(tr_names + te_names, tr_labels + te_labels)
        coord_cache = build_coord_cache(tr_paths + te_paths, feature_root=feature_root, coord_root=coord_root)
    elif args.dataset == "cm16":
        paths, labels, names, class_names = collect_cm16(args.feature_root, split="all", raw_root=args.cm16_raw_root)
        label_map = build_label_map(names, labels)
        coord_cache = build_coord_cache(paths, feature_root=feature_root, coord_root=coord_root)
    elif args.dataset == "nsclc":
        paths, labels, names, class_names = collect_nsclc(args.feature_root)
        label_map = build_label_map(names, labels)
        coord_cache = build_coord_cache(paths, feature_root=feature_root, coord_root=coord_root)
    elif args.dataset == "panda":
        paths, labels, names, class_names = collect_panda(args.feature_root, args.label_csv)
        label_map = build_label_map(names, labels)
        coord_cache = build_coord_cache(paths, feature_root=feature_root, coord_root=coord_root)
    elif fixed_split:
        tr_paths, tr_labels, tr_names, class_names = collect_cm17(args.feature_root, args.train_label_csv)
        te_paths, te_labels, te_names, _ = collect_cm17(args.feature_root, args.test_label_csv)
        label_map = build_label_map(tr_names + te_names, tr_labels + te_labels)
        coord_cache = build_coord_cache(tr_paths + te_paths, feature_root=feature_root, coord_root=coord_root)
    else:
        paths, labels, names, class_names = collect_cm17(args.feature_root, args.label_csv)
        label_map = build_label_map(names, labels)
        coord_cache = build_coord_cache(paths, feature_root=feature_root, coord_root=coord_root)

    if not coord_cache:
        raise RuntimeError(
            f"No coordinate files were matched under coord_root={coord_root} "
            f"for feature_root={feature_root}. Check the feature/coord directory pairing."
        )
    custom_index_cache = build_custom_index_cache(
        tr_paths + te_paths if (fixed_split or cm16_official_split) else paths,
        feature_root=feature_root,
        custom_index_root=custom_index_root,
    )
    if normalize_sampling_mode(args.sampling_mode) == "custom" and not custom_index_cache:
        raise RuntimeError(
            "sampling_mode=custom requires matching custom index .npy files under "
            f"custom_index_root={custom_index_root}"
        )

    print(f"\n{'=' * 68}")
    print("  TITAN subsample eval")
    print(f"  Dataset       : {args.dataset.upper()}")
    print(f"  Patch ratio   : {args.patch_ratio:.3f}")
    print(f"  Train ratio   : {args.train_patch_ratio:.3f}")
    print(f"  Eval seeds    : {args.n_eval_seeds}")
    print(f"  Sampling mode : {normalize_sampling_mode(args.sampling_mode)}")
    print(f"  Device        : {device}")
    print(f"  Coord cache   : {len(coord_cache)} slides")
    print(f"{'=' * 68}")

    titan = load_titan(model_dir=model_dir, device=device)

    summary = {
        "dataset": args.dataset,
        "patch_ratio": args.patch_ratio,
        "train_patch_ratio": args.train_patch_ratio,
        "sampling_mode": normalize_sampling_mode(args.sampling_mode),
        "n_eval_seeds": args.n_eval_seeds,
        "seed": args.seed,
        "class_names": class_names,
        "cache_dir": str(cache_dir),
        "cm16_split_mode": args.cm16_split_mode if args.dataset == "cm16" else None,
        "custom_index_root": str(custom_index_root) if custom_index_root is not None else None,
        "results": {},
    }
    train_ratio_tag = f"train{int(round(args.train_patch_ratio * 100)):03d}pct"

    if cm16_official_split or fixed_split:
        train_cache = cache_dir / f"titan_train_embs_fixed_{train_ratio_tag}.npz"
        probe_cache = cache_dir / f"titan_probe_fixed_{train_ratio_tag}.pkl"
        train_embs = None
        if probe_cache.exists():
            print(f"    loaded cached probe -> {probe_cache.name}")
            saved_probe = joblib.load(probe_cache)
        else:
            train_embs, train_lbls, train_slide_ids = load_or_extract_train_embeddings(
                cache_path=train_cache,
                titan=titan,
                train_paths=tr_paths,
                label_map=label_map,
                coord_root=coord_root,
                patch_ratio=args.train_patch_ratio,
                device=device,
                seed=args.seed,
                coord_cache=coord_cache,
                titan_slide_feat_dir=titan_slide_feat_dir,
                custom_index_cache=custom_index_cache,
            )
            saved_probe = load_or_fit_probe(
                probe_path=probe_cache,
                train_embs=train_embs,
                train_labels=train_lbls,
                seed=args.seed,
                max_iter=args.probe_max_iter,
                require_cached_probe=args.require_cached_probe,
            )

        seed_runs = []
        print("\n  [TITAN]")
        for eval_seed in range(args.seed, args.seed + args.n_eval_seeds):
            run = evaluate_with_probe(
                titan=titan,
                saved_probe=saved_probe,
                eval_paths=te_paths,
                label_map=label_map,
                coord_root=coord_root,
                patch_ratio=args.patch_ratio,
                device=device,
                seed=eval_seed,
                sampling_mode=args.sampling_mode,
                fix_alibi=args.fix_alibi,
                coord_cache=coord_cache,
                custom_index_cache=custom_index_cache,
            )
            seed_runs.append(run)
            print(
                f"    seed={eval_seed}  acc={run.get('acc', float('nan')):.4f}  "
                f"macro_f1={run.get('macro_f1', float('nan')):.4f}  "
                f"auroc={run.get('auroc', float('nan')):.4f}  "
                f"avg_patches={run.get('avg_patches', float('nan')):.1f}  "
                f"latency={run.get('latency', float('nan')):.2f}s  "
                f"latency/slide={run.get('latency_per_slide', float('nan')):.4f}s"
            )

        agg = aggregate_runs(seed_runs)
        print_aggregate("aggregate", agg)
        summary["split"] = "official_train_test" if cm16_official_split else "fixed"
        summary["n_train"] = len(tr_paths)
        summary["n_test"] = len(te_paths)
        summary["results"]["titan"] = {
            "train_cache": str(train_cache),
            "probe_cache": str(probe_cache),
            "n_train_embs": int(len(train_embs)) if train_embs is not None else None,
            "seed_runs": seed_runs,
            "aggregate": agg,
        }
    else:
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
        fold_means = []
        fold_details = []

        print("\n  [TITAN]")
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            tr_paths_fold = [paths[i] for i in train_idx]
            va_paths_fold = [paths[i] for i in val_idx]

            train_cache = cache_dir / f"titan_train_embs_fold{fold_idx}_{train_ratio_tag}.npz"
            probe_cache = cache_dir / f"titan_probe_fold{fold_idx}_{train_ratio_tag}.pkl"

            print(f"    fold={fold_idx}")
            train_embs = None
            if probe_cache.exists():
                print(f"      loaded cached probe -> {probe_cache.name}")
                saved_probe = joblib.load(probe_cache)
            else:
                train_embs, train_lbls, _ = load_or_extract_train_embeddings(
                    cache_path=train_cache,
                    titan=titan,
                    train_paths=tr_paths_fold,
                    label_map=label_map,
                    coord_root=coord_root,
                    patch_ratio=args.train_patch_ratio,
                    device=device,
                    seed=args.seed + fold_idx,
                    coord_cache=coord_cache,
                    titan_slide_feat_dir=titan_slide_feat_dir,
                    custom_index_cache=custom_index_cache,
                )
                saved_probe = load_or_fit_probe(
                    probe_path=probe_cache,
                    train_embs=train_embs,
                    train_labels=train_lbls,
                    seed=args.seed + fold_idx,
                    max_iter=args.probe_max_iter,
                    require_cached_probe=args.require_cached_probe,
                )

            seed_runs = []
            for eval_seed in range(args.seed, args.seed + args.n_eval_seeds):
                run = evaluate_with_probe(
                    titan=titan,
                    saved_probe=saved_probe,
                    eval_paths=va_paths_fold,
                    label_map=label_map,
                    coord_root=coord_root,
                    patch_ratio=args.patch_ratio,
                    device=device,
                    seed=eval_seed,
                    sampling_mode=args.sampling_mode,
                    fix_alibi=args.fix_alibi,
                    coord_cache=coord_cache,
                    custom_index_cache=custom_index_cache,
                )
                seed_runs.append(run)
                print(
                    f"      seed={eval_seed}  acc={run.get('acc', float('nan')):.4f}  "
                    f"macro_f1={run.get('macro_f1', float('nan')):.4f}  "
                    f"auroc={run.get('auroc', float('nan')):.4f}  "
                    f"avg_patches={run.get('avg_patches', float('nan')):.1f}  "
                    f"latency={run.get('latency', float('nan')):.2f}s  "
                    f"latency/slide={run.get('latency_per_slide', float('nan')):.4f}s"
                )

            fold_agg = aggregate_runs(seed_runs)
            fold_mean = fold_agg["mean"]
            fold_means.append(fold_mean)
            fold_details.append(
                {
                    "fold": fold_idx,
                    "train_cache": str(train_cache),
                    "probe_cache": str(probe_cache),
                    "n_train_embs": int(len(train_embs)) if train_embs is not None else None,
                    "n_val": len(va_paths_fold),
                    "seed_runs": seed_runs,
                    "aggregate": fold_agg,
                }
            )
            print_aggregate(f"fold {fold_idx}", fold_agg)

        overall = aggregate_runs(fold_means)
        print_aggregate("overall", overall)
        summary["split"] = split_desc
        summary["split_file"] = str(split_path)
        summary["split_dir"] = str(split_path.with_suffix(""))
        summary["n_samples"] = len(paths)
        summary["test_size"] = args.test_size
        summary["results"]["titan"] = {
            "folds": fold_details,
            "aggregate": overall,
        }

    summary_path = out_dir / "subsample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved -> {summary_path}")


if __name__ == "__main__":
    main()
