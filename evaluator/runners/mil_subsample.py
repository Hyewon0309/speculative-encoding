"""
Evaluate saved MIL checkpoints with test-time patch subsampling.

This script is intended for cases where checkpoints were already trained with
all patches, and only inference should use a reduced subset such as 5%.

Supported modes:
  - cm16: evaluate saved fold checkpoints on shared stratified 8:2 splits over all slides
  - cm17: evaluate saved fold checkpoints on shared stratified 8:2 splits
  - panda: shared repeated stratified 8:2 splits using saved fold checkpoints
  - nsclc: shared patient-level stratified 8:2 splits using saved fold checkpoints
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# REPO_ROOT = speculative_encoding/ (two levels above this file).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.runners.custom_index_utils import build_custom_index_cache
from evaluator.runners.feasibility_subsample import find_coord_file, load_coords, subsample_indices
from evaluator.mil import build_net, evaluate as mil_evaluate, make_conf, set_seed
from evaluator.runners.mil_comparison import (
    ARCH_DISPLAY,
    FEATURE_DIM,
    ALL_MIL_ARCHS,
    CM16_RAW_ROOT,
    collect_brca,
    collect_cm16,
    collect_cm17,
    collect_nsclc,
    collect_panda,
    get_or_create_shared_splits,
    get_tcga_patient_id,
)


DEFAULT_COORD_ROOT = os.environ.get("COORD_ROOT", "")


def _cuda_sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_slide(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, dict):
        key = "features" if "features" in data else list(data.keys())[0]
        return data[key].float()
    raise TypeError(f"Unexpected .pt content: {type(data)}")


def normalize_sampling_mode(mode: str) -> str:
    mode = mode.replace("-", "_")
    return "geometric" if mode == "grid" else mode


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


def build_coord_cache(paths, feature_root: Path, coord_root: Path) -> Dict[str, Tuple[np.ndarray, int]]:
    coord_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    flat_paths = []
    for path in paths:
        if isinstance(path, list):
            flat_paths.extend(path)
        else:
            flat_paths.append(path)

    for path in flat_paths:
        slide_id = path.stem
        if slide_id in coord_cache:
            continue
        coord_path = infer_coord_path(path, feature_root=feature_root, coord_root=coord_root)
        if coord_path is None:
            continue
        coord_cache[slide_id] = load_coords(coord_path)
    return coord_cache


def subsample_features(
    features: torch.Tensor,
    patch_ratio: float,
    seed: int,
    sampling_mode: str = "random",
    coords: Optional[np.ndarray] = None,
    custom_indices: Optional[np.ndarray] = None,
    fill_features: Optional[torch.Tensor] = None,
    fill_random_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    if fill_features is None:
        if sampling_mode != "custom" and patch_ratio >= 1.0:
            return features, 0.0
        mode = normalize_sampling_mode(sampling_mode)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        keep_idx = subsample_indices(
            len(features),
            patch_ratio,
            rng,
            mode=mode,
            coords=coords,
            features=features,
            custom_indices=custom_indices,
        )
        selection_seconds = time.perf_counter() - t0
        return features[torch.from_numpy(keep_idx).long()], selection_seconds

    if fill_features.shape != features.shape:
        raise ValueError(
            f"fill_features shape {tuple(fill_features.shape)} does not match features "
            f"{tuple(features.shape)}"
    )
    mode = normalize_sampling_mode(sampling_mode)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    keep_idx = subsample_indices(
        len(features),
        patch_ratio,
        rng,
        mode=mode,
        coords=coords,
        features=features,
        custom_indices=custom_indices,
    )
    selection_seconds = time.perf_counter() - t0
    keep_t = torch.from_numpy(keep_idx).long()

    if fill_random_ratio is None or fill_random_ratio <= 0.0:
        out = fill_features.clone()
        out[keep_t] = features[keep_t]
        return out, selection_seconds

    N = int(features.shape[0])
    mask = np.ones(N, dtype=bool)
    mask[keep_idx] = False
    non_selected = np.arange(N)[mask]
    k_fill = int(round(float(fill_random_ratio) * N))
    k_fill = min(k_fill, int(non_selected.size))
    if k_fill <= 0:
        return features[keep_t], selection_seconds
    fill_pick = rng.choice(non_selected, size=k_fill, replace=False)
    fill_pick.sort()
    fill_t = torch.from_numpy(fill_pick).long()
    return torch.cat([features[keep_t], fill_features[fill_t]], dim=0), selection_seconds


def _resolve_fill_path(
    feature_path: Path,
    feature_root: Optional[Path],
    fill_root: Path,
) -> Path:
    if feature_root is not None:
        try:
            rel = feature_path.relative_to(feature_root)
            return fill_root / rel
        except ValueError:
            pass
    return fill_root / feature_path.name


class SubsampledSlideDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        patch_ratio: float = 1.0,
        seed: int = 42,
        sampling_mode: str = "random",
        coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
        custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
        feature_root: Optional[Path] = None,
        fill_feature_root: Optional[Path] = None,
        fill_random_ratio: Optional[float] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.patch_ratio = patch_ratio
        self.seed = seed
        self.sampling_mode = normalize_sampling_mode(sampling_mode)
        self.coord_cache = coord_cache or {}
        self.custom_index_cache = custom_index_cache or {}
        self.feature_root = feature_root
        self.fill_feature_root = fill_feature_root
        self.fill_random_ratio = fill_random_ratio

    def __len__(self):
        return len(self.paths)

    def _load_fill(self, path: Path) -> Optional[torch.Tensor]:
        if self.fill_feature_root is None:
            return None
        fill_path = _resolve_fill_path(path, self.feature_root, self.fill_feature_root)
        if not fill_path.exists():
            raise FileNotFoundError(
                f"fill feature file not found for slide {path.stem}: {fill_path}"
            )
        return load_slide(fill_path)

    def __getitem__(self, idx):
        path = self.paths[idx]
        selection_seconds = 0.0
        if isinstance(path, list):
            feats = []
            for sub_idx, sub_path in enumerate(path):
                feat = load_slide(sub_path)
                coords = self.coord_cache.get(sub_path.stem, (None, None))[0]
                custom_indices = self.custom_index_cache.get(sub_path.stem)
                fill_features = self._load_fill(sub_path)
                feat, feat_selection_seconds = subsample_features(
                    feat,
                    self.patch_ratio,
                    seed=self.seed + idx * 100003 + sub_idx,
                    sampling_mode=self.sampling_mode,
                    coords=coords,
                    custom_indices=custom_indices,
                    fill_features=fill_features,
                    fill_random_ratio=self.fill_random_ratio,
                )
                selection_seconds += feat_selection_seconds
                feats.append(feat)
            feat = torch.cat(feats, dim=0)
        else:
            feat = load_slide(path)
            coords = self.coord_cache.get(path.stem, (None, None))[0]
            custom_indices = self.custom_index_cache.get(path.stem)
            fill_features = self._load_fill(path)
            feat, selection_seconds = subsample_features(
                feat,
                self.patch_ratio,
                seed=self.seed + idx,
                sampling_mode=self.sampling_mode,
                coords=coords,
                custom_indices=custom_indices,
                fill_features=fill_features,
            )

        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        return {
            "input": feat.unsqueeze(0),
            "label": label,
            "coords": torch.zeros(1, feat.shape[0], 2),
            "selection_time": torch.tensor([selection_seconds], dtype=torch.float32),
        }


def _collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def make_loader(
    paths,
    labels,
    patch_ratio: float = 1.0,
    seed: int = 42,
    sampling_mode: str = "random",
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
    feature_root: Optional[Path] = None,
    fill_feature_root: Optional[Path] = None,
    fill_random_ratio: Optional[float] = None,
):
    dataset = SubsampledSlideDataset(
        paths,
        labels,
        patch_ratio=patch_ratio,
        seed=seed,
        sampling_mode=sampling_mode,
        coord_cache=coord_cache,
        custom_index_cache=custom_index_cache,
        feature_root=feature_root,
        fill_feature_root=fill_feature_root,
        fill_random_ratio=fill_random_ratio,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_one,
    )


def metric_dict(
    auroc: float,
    acc: float,
    f1: float,
    loss: float,
    precision: float,
    recall: float,
) -> Dict[str, float]:
    return {
        "acc":       float(acc / 100.0 if acc > 1.0 else acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "auroc":     float(auroc),
        "loss":      float(loss),
    }


def aggregate_runs(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        return {"mean": {}, "std": {}}

    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in runs if key in r and not math.isnan(r[key])]
        if vals:
            mean[key] = float(np.mean(vals))
            std[key] = float(np.std(vals))
    return {"mean": mean, "std": std}


def print_aggregate(label: str, agg: Dict[str, Dict[str, float]]) -> None:
    mean = agg.get("mean", {})
    std = agg.get("std", {})
    metrics = ["acc", "precision", "recall", "f1", "auroc", "loss", "latency", "latency_per_slide"]
    parts = []
    for metric in metrics:
        if metric in mean:
            parts.append(f"{metric}={mean[metric]:.4f}±{std.get(metric, 0.0):.4f}")
    print(f"    {label}: " + "  ".join(parts))


def evaluate_checkpoint(
    arch: str,
    ckpt_path: Path,
    eval_paths,
    eval_labels,
    n_class: int,
    feature_dim: Optional[int],
    device: str,
    train_epoch: int,
    lr: float,
    wd: float,
    eval_interval: int,
    patch_ratio: float,
    seed: int,
    sampling_mode: str,
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
    feature_root: Optional[Path] = None,
    fill_feature_root: Optional[Path] = None,
    fill_random_ratio: Optional[float] = None,
) -> Dict[str, float]:
    set_seed(seed)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resolved_feature_dim = int(ckpt.get("feature_dim", feature_dim or FEATURE_DIM))
    conf = make_conf(
        arch=arch,
        feature_dim=resolved_feature_dim,
        n_class=n_class,
        train_epoch=train_epoch,
        lr=lr,
        wd=wd,
        eval_interval=eval_interval,
    )
    net = build_net(conf, torch.device(device))
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    net.load_state_dict(state)
    net = net.to(device).eval()

    criterion = nn.CrossEntropyLoss()
    loader = make_loader(
        eval_paths,
        eval_labels,
        patch_ratio=patch_ratio,
        seed=seed,
        sampling_mode=sampling_mode,
        coord_cache=coord_cache,
        custom_index_cache=custom_index_cache,
        feature_root=feature_root,
        fill_feature_root=fill_feature_root,
        fill_random_ratio=fill_random_ratio,
    )
    eval_out = mil_evaluate(
        net,
        criterion,
        loader,
        torch.device(device),
        conf,
        header=f"{arch} eval",
        return_details=True,
        measure_gpu_time=True,
    )
    (
        auroc,
        acc,
        f1,
        loss,
        precision,
        recall,
        detail_metrics,
        _,
        _,
        *timing_out,
    ) = eval_out
    if len(timing_out) == 2:
        gpu_seconds, selection_seconds = timing_out
    else:
        gpu_seconds, selection_seconds = 0.0, 0.0
    metrics = metric_dict(
        auroc,
        acc,
        f1,
        loss,
        precision,
        recall,
    )
    metrics["selection_time"] = float(selection_seconds)
    metrics["gpu_time"] = float(gpu_seconds)
    metrics["latency"] = float(gpu_seconds + selection_seconds)
    metrics["latency_per_slide"] = float((gpu_seconds + selection_seconds) / len(eval_labels)) if len(eval_labels) > 0 else float("nan")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved MIL checkpoints with patch subsampling",
    )
    parser.add_argument("--dataset", required=True, choices=["brca", "cm16", "cm17", "panda", "nsclc"])
    parser.add_argument("--feature_root", default=os.environ.get("FEATURE_ROOT", ""),
                        help="Patch-feature root. Defaults to $FEATURE_ROOT.")
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--coord_root", default=DEFAULT_COORD_ROOT)
    parser.add_argument("--cm16_split_mode", default="cv", choices=["cv", "official"])
    parser.add_argument("--cm16_raw_root", default=CM16_RAW_ROOT)
    parser.add_argument("--label_csv", default=os.environ.get("CM17_LABEL_CSV") or None,
                        help="CM17 stages.csv. Defaults to $CM17_LABEL_CSV.")
    parser.add_argument("--train_label_csv", default=None)
    parser.add_argument("--test_label_csv", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--mil_archs", nargs="+", default=["abmil"], choices=ALL_MIL_ARCHS)
    parser.add_argument("--train_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--patch_ratio", type=float, default=0.05)
    parser.add_argument(
        "--sampling_mode",
        default="random",
        choices=[
            "random",
            "grid",
            "geometric",
            "k_means",
            "k_medoid",
            "k-medoid",
            "hdbscan",
            "custom",
            "custom_inverse",
        ],
    )
    parser.add_argument("--n_eval_seeds", type=int, default=1)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--checkpoint_fold",
        type=int,
        default=None,
        help="If set, official-split eval loads <arch>_fold{N}_best.pt (single-fold shortcut).",
    )
    parser.add_argument(
        "--checkpoint_folds",
        type=int,
        nargs="+",
        default=None,
        help="List of fold indices; official-split eval averages over folds (<arch>_fold{N}_best.pt).",
    )
    parser.add_argument("--custom_index_root", default=None)
    parser.add_argument(
        "--fill_feature_root",
        default=None,
        help=(
            "Optional feature root used to fill NON-selected (non-custom-index) patch "
            "positions. When set with sampling_mode=custom, the loader returns a "
            "full-length (N, D) tensor where selected positions come from --feature_root "
            "and all other positions come from --fill_feature_root. Shapes must match "
            "per slide. Default None = existing behavior (subsample only)."
        ),
    )
    parser.add_argument(
        "--fill_random_ratio",
        type=float,
        default=None,
        help=(
            "When set together with --fill_feature_root, fill only round(ratio * N) "
            "randomly-chosen positions from the non-selected set (disjoint from the "
            "custom-index selection). Remaining non-selected positions are dropped "
            "from the bag. Final bag length = |custom| + round(ratio * N). "
            "Default None = fill all non-selected positions."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_root = Path(args.feature_root)
    coord_root = Path(args.coord_root)
    custom_index_root = Path(args.custom_index_root) if args.custom_index_root else None
    fill_feature_root = Path(args.fill_feature_root) if args.fill_feature_root else None
    fill_random_ratio = float(args.fill_random_ratio) if args.fill_random_ratio is not None else None

    if args.output_dir is None:
        ratio_tag = f"{int(round(args.patch_ratio * 100)):02d}pct"
        args.output_dir = f"./results/{args.dataset}_mil_subsample_{ratio_tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    fixed_split = args.train_label_csv is not None and args.test_label_csv is not None
    cm16_official_split = args.dataset == "cm16" and args.cm16_split_mode == "official"

    if args.dataset == "brca":
        paths, labels, names, class_names = collect_brca(args.feature_root)
    elif cm16_official_split:
        tr_paths, tr_labels, tr_names, class_names = collect_cm16(
            args.feature_root, split="train", raw_root=args.cm16_raw_root
        )
        te_paths, te_labels, te_names, _ = collect_cm16(
            args.feature_root, split="test", raw_root=args.cm16_raw_root
        )
    elif args.dataset == "cm16":
        paths, labels, names, class_names = collect_cm16(args.feature_root, split="all", raw_root=args.cm16_raw_root)
    elif args.dataset == "nsclc":
        paths, labels, names, class_names = collect_nsclc(args.feature_root)
    elif args.dataset == "panda":
        paths, labels, names, class_names = collect_panda(args.feature_root, args.label_csv)
    elif fixed_split:
        tr_paths, tr_labels, tr_names, class_names = collect_cm17(args.feature_root, args.train_label_csv)
        te_paths, te_labels, te_names, _ = collect_cm17(args.feature_root, args.test_label_csv)
    else:
        paths, labels, names, class_names = collect_cm17(args.feature_root, args.label_csv)

    if fixed_split or cm16_official_split:
        coord_cache = build_coord_cache(tr_paths + te_paths, feature_root=feature_root, coord_root=coord_root)
    else:
        coord_cache = build_coord_cache(paths, feature_root=feature_root, coord_root=coord_root)
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
    print(f"  Saved-checkpoint subsample eval")
    print(f"  Dataset     : {args.dataset.upper()}")
    print(f"  Patch ratio : {args.patch_ratio:.3f}")
    print(f"  Sampling    : {normalize_sampling_mode(args.sampling_mode)}")
    print(f"  Eval seeds  : {args.n_eval_seeds}")
    print(f"  Methods     : {[ARCH_DISPLAY[a] for a in args.mil_archs]}")
    print(f"  Device      : {device}")
    print(f"  Checkpoints : {checkpoint_dir}")
    print(f"{'=' * 68}")

    summary = {
        "dataset": args.dataset,
        "feature": Path(args.feature_root).name,
        "feature_root": args.feature_root,
        "feature_dim": args.feature_dim,
        "patch_ratio": args.patch_ratio,
        "sampling_mode": normalize_sampling_mode(args.sampling_mode),
        "n_eval_seeds": args.n_eval_seeds,
        "seed": args.seed,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "cm16_split_mode": args.cm16_split_mode if args.dataset == "cm16" else None,
        "custom_index_root": str(custom_index_root) if custom_index_root is not None else None,
        "fill_feature_root": str(fill_feature_root) if fill_feature_root is not None else None,
        "fill_random_ratio": fill_random_ratio,
        "mil_archs": args.mil_archs,
        "results": {},
    }

    if fixed_split or cm16_official_split:
        n_class = len(set(tr_labels + te_labels))
        summary["split"] = "official_train_test" if cm16_official_split else "fixed"
        summary["n_train"] = len(tr_paths)
        summary["n_test"] = len(te_paths)
        summary["class_names"] = class_names

        if args.checkpoint_folds is not None:
            folds_list = list(args.checkpoint_folds)
        elif args.checkpoint_fold is not None:
            folds_list = [args.checkpoint_fold]
        else:
            folds_list = None  # legacy {arch}_best.pt single-checkpoint path
        summary["checkpoint_folds"] = folds_list

        for arch in args.mil_archs:
            print(f"\n  [{ARCH_DISPLAY[arch]}]")

            if folds_list is None:
                ckpt_path = checkpoint_path if checkpoint_path is not None else (checkpoint_dir / f"{arch}_best.pt")
                iter_folds = [(None, ckpt_path)]
            else:
                iter_folds = [
                    (fold_idx, checkpoint_dir / f"{arch}_fold{fold_idx}_best.pt")
                    for fold_idx in folds_list
                ]

            fold_details = []
            fold_means = []
            for fold_idx, ckpt_path in iter_folds:
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                seed_runs = []
                for eval_seed in range(args.seed, args.seed + args.n_eval_seeds):
                    run = evaluate_checkpoint(
                        arch=arch,
                        ckpt_path=ckpt_path,
                        eval_paths=te_paths,
                        eval_labels=te_labels,
                        n_class=n_class,
                        feature_dim=args.feature_dim,
                        device=device,
                        train_epoch=args.train_epoch,
                        lr=args.lr,
                        wd=args.wd,
                        eval_interval=args.eval_interval,
                        patch_ratio=args.patch_ratio,
                        seed=eval_seed,
                        sampling_mode=args.sampling_mode,
                        coord_cache=coord_cache,
                        custom_index_cache=custom_index_cache,
                        feature_root=feature_root,
                        fill_feature_root=fill_feature_root,
                        fill_random_ratio=fill_random_ratio,
                    )
                    seed_runs.append(run)
                    prefix = f"fold={fold_idx} " if fold_idx is not None else ""
                    print(
                        f"    {prefix}seed={eval_seed}  acc={run['acc']:.4f}  "
                        f"f1={run['f1']:.4f}  auroc={run['auroc']:.4f}"
                    )

                fold_agg = aggregate_runs(seed_runs)
                fold_means.append(fold_agg["mean"])
                fold_details.append(
                    {
                        "fold": fold_idx,
                        "checkpoint": str(ckpt_path),
                        "seed_runs": seed_runs,
                        "aggregate": fold_agg,
                    }
                )
                if fold_idx is not None:
                    print_aggregate(f"fold {fold_idx}", fold_agg)

            overall = aggregate_runs(fold_means)
            label = "overall" if folds_list is not None else "aggregate"
            print_aggregate(label, overall)
            if folds_list is not None:
                summary["results"][arch] = {
                    "folds": fold_details,
                    "aggregate": overall,
                }
            else:
                summary["results"][arch] = {
                    "seed_runs": fold_details[0]["seed_runs"],
                    "aggregate": overall,
                }
    else:
        n_class = len(set(labels))
        if args.dataset in {"brca", "cm16", "cm17", "panda", "nsclc"}:
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
        else:
            raise ValueError(f"Shared split evaluation is not supported for dataset={args.dataset}")

        summary["split"] = split_desc
        summary["split_file"] = str(split_path)
        summary["split_dir"] = str(split_path.with_suffix(""))
        summary["n_samples"] = len(paths)
        summary["test_size"] = args.test_size
        if args.dataset == "nsclc":
            summary["n_patients"] = len({get_tcga_patient_id(name) for name in names})
        summary["class_names"] = class_names

        for arch in args.mil_archs:
            print(f"\n  [{ARCH_DISPLAY[arch]}]")
            fold_means = []
            fold_details = []

            for fold_idx, (_, val_idx) in enumerate(folds):
                ckpt_path = checkpoint_dir / f"{arch}_fold{fold_idx}_best.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                val_paths = [paths[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]

                seed_runs = []
                print(f"    fold={fold_idx}")
                for eval_seed in range(args.seed, args.seed + args.n_eval_seeds):
                    run = evaluate_checkpoint(
                        arch=arch,
                        ckpt_path=ckpt_path,
                        eval_paths=val_paths,
                        eval_labels=val_labels,
                        n_class=n_class,
                        feature_dim=args.feature_dim,
                        device=device,
                        train_epoch=args.train_epoch,
                        lr=args.lr,
                        wd=args.wd,
                        eval_interval=args.eval_interval,
                        patch_ratio=args.patch_ratio,
                        seed=eval_seed,
                        sampling_mode=args.sampling_mode,
                        coord_cache=coord_cache,
                        custom_index_cache=custom_index_cache,
                        feature_root=feature_root,
                        fill_feature_root=fill_feature_root,
                        fill_random_ratio=fill_random_ratio,
                    )
                    seed_runs.append(run)
                    print(
                        f"      seed={eval_seed}  acc={run['acc']:.4f}  "
                        f"f1={run['f1']:.4f}  auroc={run['auroc']:.4f}"
                    )

                fold_agg = aggregate_runs(seed_runs)
                fold_mean = fold_agg["mean"]
                fold_means.append(fold_mean)
                fold_details.append(
                    {
                        "fold": fold_idx,
                        "checkpoint": str(ckpt_path),
                        "n_val": len(val_paths),
                        "seed_runs": seed_runs,
                        "aggregate": fold_agg,
                    }
                )
                print_aggregate(f"fold {fold_idx}", fold_agg)

            overall = aggregate_runs(fold_means)
            print_aggregate("overall", overall)
            summary["results"][arch] = {
                "folds": fold_details,
                "aggregate": overall,
            }

    summary_path = out_dir / "subsample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved -> {summary_path}")


if __name__ == "__main__":
    main()
