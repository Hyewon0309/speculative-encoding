"""Shared sampler / slide-encoder helpers used by the per-model runners.

This module exposes:

* ``load_features`` — load a per-slide ``.pt`` patch-feature tensor.
* ``find_coord_file`` / ``load_coords`` — discover & parse coord ``.npy`` files.
* ``subsample_indices`` — return the patch indices kept by a sampler:
    ``random``, ``geometric`` (grid), ``k_means`` / ``k_medoid`` / ``hdbscan``
    (RAPIDS-accelerated), ``custom`` / ``custom_inverse`` (pre-computed).
* ``titan_extract_embeddings`` — run TITAN on a list of slides at a given
  patch budget and return per-slide ``[D]`` embeddings.
* ``gigapath_extract_embeddings`` — same for Prov-GigaPath.

Imported by ``mil_subsample.py``, ``titan_subsample.py``,
``gigapath_subsample.py``.
"""

import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
# REPO_ROOT is the speculative_encoding/ folder. ``load_paths.sh`` prepends
# it to PYTHONPATH already; we re-add it so the module also works when one
# of the runners is invoked as a plain ``python evaluator/runners/<x>.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defaults read from env. Pass explicit --feature_root / --coord_root on the
# CLI to override per call. Empty string means "must be supplied explicitly".
# (TCGA-specific defaults previously lived here; the per-runner CLIs now
# carry their own --feature_root / --coord_root + env-var fallbacks.)

# Whether we have already warned about the RAPIDS k-means fallback path.
_KMEANS_RAPIDS_FALLBACK_WARNED = False


# ── Data loading helpers ────────────────────────────────────────────────────

def load_features(path: Path) -> torch.Tensor:
    """Load ``[N, D]`` patch-feature tensor from a ``.pt`` file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        key = "features" if "features" in data else next(iter(data))
        feat = data[key]
    else:
        feat = data
    return feat.float()


def find_coord_file(coord_dir: Path, cancer_type: str, slide_id: str) -> Optional[Path]:
    """Find the coord ``.npy`` for ``slide_id`` under ``coord_dir/cancer_type/``."""
    exact = coord_dir / cancer_type / f"{slide_id}.npy"
    if exact.exists():
        return exact
    matches = list((coord_dir / cancer_type).glob(f"{slide_id}.*.npy"))
    return matches[0] if matches else None


def load_coords(coord_path: Path) -> Tuple[np.ndarray, int]:
    """Load patch coordinates from a structured-array ``.npy``.

    Returns
    -------
    coords         : ``np.ndarray [N, 2]`` (x, y in level-0 pixel space)
    tile_size_lv0  : int
    """
    arr = np.load(coord_path, allow_pickle=True)
    coords = np.stack([arr["x"], arr["y"]], axis=1).astype(np.int64)
    tile_size_lv0 = int(arr["tile_size_lv0"][0])
    return coords, tile_size_lv0


# ── Sampler helpers ─────────────────────────────────────────────────────────

def _ensure_target_size(
    selected: List[int],
    N: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deduplicate ``selected`` and fill/truncate to exactly ``k`` indices."""
    ordered_unique: List[int] = []
    seen = set()
    for idx in selected:
        idx_i = int(idx)
        if idx_i not in seen:
            ordered_unique.append(idx_i)
            seen.add(idx_i)

    if len(ordered_unique) > k:
        ordered_unique = ordered_unique[:k]

    if len(ordered_unique) < k:
        remaining = np.setdiff1d(np.arange(N), np.array(ordered_unique, dtype=np.int64))
        extra = min(k - len(ordered_unique), len(remaining))
        if extra > 0:
            ordered_unique.extend(rng.choice(remaining, extra, replace=False).tolist())

    return np.sort(np.array(ordered_unique, dtype=np.int64))


def _import_rapids_cluster_modules():
    """Lazily import RAPIDS modules only when fast cluster sampling is requested."""
    try:
        import cupy as cp
        from cuml.cluster import HDBSCAN as CuMLHDBSCAN
        from cuml.cluster import KMeans as CuMLKMeans
        return cp, CuMLKMeans, CuMLHDBSCAN
    except Exception as exc:
        raise RuntimeError(
            "RAPIDS sampling requires cupy + cuml. "
            "Install via `pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 cupy-cuda12x`."
        ) from exc


def _to_rapids_feature_matrix(features: torch.Tensor):
    """Move ``features`` to a CUDA torch tensor and return a CuPy zero-copy view."""
    cp, _, _ = _import_rapids_cluster_modules()
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if not torch.cuda.is_available():
        raise RuntimeError("RAPIDS cluster sampling requires CUDA.")
    feats_t = features.to(device="cuda", dtype=torch.float32).contiguous()
    feats_cp = cp.from_dlpack(feats_t.detach())
    return feats_t, feats_cp


def _select_min_distance_representatives(labels, sq_dists) -> np.ndarray:
    """Pick one point per cluster — the one closest to its assigned centroid.

    Vectorised: sort rows by ``(label, distance)``, keep the first row of
    every label segment.
    """
    cp, _, _ = _import_rapids_cluster_modules()
    if int(labels.size) == 0:
        return np.empty(0, dtype=np.int64)
    labels_i = labels.astype(cp.int32, copy=False)
    sort_keys = cp.stack((sq_dists, labels_i), axis=0)
    order = cp.lexsort(sort_keys)
    ordered_labels = labels_i[order]
    first_in_group = cp.empty(ordered_labels.shape, dtype=cp.bool_)
    first_in_group[0] = True
    if int(ordered_labels.size) > 1:
        first_in_group[1:] = ordered_labels[1:] != ordered_labels[:-1]
    selected = order[first_in_group]
    return np.sort(cp.asnumpy(selected).astype(np.int64, copy=False))


def subsample_indices(
    N: int,
    patch_ratio: float,
    rng: np.random.Generator,
    mode: str = "random",
    coords: Optional[np.ndarray] = None,
    features: Optional[torch.Tensor] = None,
    custom_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return sorted indices keeping ``patch_ratio`` fraction of ``N`` patches.

    Modes
    -----
    ``random``         uniform random sampling (default).
    ``geometric``      spatially-uniform grid sampling — needs ``coords [N, 2]``.
    ``k_means``        RAPIDS cuML KMeans on features, keep nearest to centroid.
    ``k_medoid``       k-medoids-style representatives on features.
    ``hdbscan``        RAPIDS cuML HDBSCAN, one representative per dense cluster.
    ``custom``         use externally provided ``custom_indices``.
    ``custom_inverse`` use every index *except* those in ``custom_indices``.

    ``patch_ratio >= 1.0`` returns ``arange(N)``.
    """
    if mode == "custom":
        if custom_indices is None:
            raise ValueError("custom sampling mode requires custom_indices")
        idx = np.asarray(custom_indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < N)]
        if idx.size == 0:
            raise ValueError("custom_indices is empty after bounds check")
        return idx
    if mode == "custom_inverse":
        if custom_indices is None:
            raise ValueError("custom_inverse sampling mode requires custom_indices")
        all_indices = np.arange(N)
        exclude_idx = np.asarray(custom_indices, dtype=np.int64).reshape(-1)
        exclude_idx = exclude_idx[(exclude_idx >= 0) & (exclude_idx < N)]
        idx = np.setdiff1d(all_indices, exclude_idx)
        if idx.size == 0:
            raise ValueError("custom_inverse result is empty (all indices were excluded)")
        return idx

    if patch_ratio >= 1.0:
        return np.arange(N)
    if mode == "k_means" and features is not None and len(features) == N:
        return _subsample_kmeans(features, patch_ratio, rng)
    if mode == "k_medoid" and features is not None and len(features) == N:
        return _subsample_kmedoid(features, patch_ratio, rng)
    if mode == "hdbscan" and features is not None and len(features) == N:
        return _subsample_hdbscan(features, patch_ratio, rng)
    if mode == "geometric" and coords is not None and len(coords) == N:
        return _subsample_geometric(coords, patch_ratio, rng)
    k = max(1, int(round(N * patch_ratio)))
    return np.sort(rng.choice(N, k, replace=False))


def _subsample_geometric(
    coords: np.ndarray,
    patch_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grid-based spatially-uniform subsampling.

    Splits the slide bounding box into a √k × √k grid and keeps one random
    patch per non-empty cell. Falls back to random sampling when the layout is
    degenerate, and tops up with random picks when fewer occupied cells exist
    than the requested ``k``.
    """
    N = len(coords)
    k = max(1, int(round(N * patch_ratio)))
    n_grid = max(1, int(np.ceil(np.sqrt(k))))

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        return np.sort(rng.choice(N, k, replace=False))

    cx = np.minimum(((x - x.min()) / x_range * n_grid).astype(int), n_grid - 1)
    cy = np.minimum(((y - y.min()) / y_range * n_grid).astype(int), n_grid - 1)
    cell_ids = cx * n_grid + cy

    cell_map: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(cell_ids.tolist()):
        cell_map[cid].append(i)
    cells = list(cell_map.keys())
    rng.shuffle(cells)

    selected: List[int] = []
    for cid in cells:
        selected.append(int(rng.choice(cell_map[cid])))
        if len(selected) >= k:
            break

    if len(selected) < k:
        remaining = list(set(range(N)) - set(selected))
        extra = min(k - len(selected), len(remaining))
        if extra > 0:
            selected.extend(rng.choice(remaining, extra, replace=False).tolist())

    return np.sort(np.array(selected, dtype=np.int64))


def _subsample_kmeans_fallback(
    features: torch.Tensor,
    patch_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """K-means fallback used when RAPIDS/cuML is not installed.

    Order of preference: ``fast_pytorch_kmeans`` → ``sklearn.cluster.KMeans``
    → uniform random (with a warning).
    """
    N = len(features)
    k = max(1, int(round(N * patch_ratio)))
    if k >= N:
        return np.arange(N)

    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    feats_cpu = features.float().detach().cpu()
    km_seed = int(rng.integers(0, 2**31))

    # 1) fast_pytorch_kmeans
    try:
        from fast_pytorch_kmeans import KMeans as FastKMeans

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feats = feats_cpu.to(device)
        torch.manual_seed(km_seed)

        kmeans = FastKMeans(n_clusters=k, mode="euclidean", max_iter=300)
        labels = kmeans.fit_predict(feats)  # [N]
        centroids = kmeans.centroids        # [k, D]

        selected = []
        for c in range(k):
            member_indices = torch.where(labels == c)[0]
            if len(member_indices) == 0:
                continue
            dists = torch.norm(feats[member_indices] - centroids[c], dim=1)
            selected.append(member_indices[torch.argmin(dists)].item())
        return _ensure_target_size(selected, N, k, rng)
    except Exception:
        pass

    # 2) sklearn KMeans
    try:
        from sklearn.cluster import KMeans as SklearnKMeans

        X = feats_cpu.numpy()
        km = SklearnKMeans(
            n_clusters=k, random_state=km_seed, n_init=10, max_iter=300, init="k-means++",
        )
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        selected = []
        for c in range(k):
            member_indices = np.where(labels == c)[0]
            if len(member_indices) == 0:
                continue
            member_feats = X[member_indices]
            dists = np.linalg.norm(member_feats - centroids[c], axis=1)
            selected.append(int(member_indices[int(np.argmin(dists))]))
        return _ensure_target_size(selected, N, k, rng)
    except Exception:
        pass

    # 3) Last resort
    warnings.warn(
        "Both fast_pytorch_kmeans and sklearn KMeans are unavailable; "
        "falling back to random sampling.",
        RuntimeWarning,
    )
    return np.sort(rng.choice(N, k, replace=False))


def _subsample_kmeans(
    features: torch.Tensor,
    patch_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """RAPIDS cuML KMeans, falling back to a CPU/torch K-means when needed."""
    global _KMEANS_RAPIDS_FALLBACK_WARNED

    N = len(features)
    k = max(1, int(round(N * patch_ratio)))
    if k >= N:
        return np.arange(N)

    try:
        cp, CuMLKMeans, _ = _import_rapids_cluster_modules()
        _, feats_cp = _to_rapids_feature_matrix(features)
        km_seed = int(rng.integers(0, 2**31))

        kmeans = CuMLKMeans(
            n_clusters=k, max_iter=300, random_state=km_seed,
            init="scalable-k-means++", output_type="cupy",
        )
        labels = cp.asarray(kmeans.fit_predict(feats_cp))
        centroids = cp.asarray(kmeans.cluster_centers_)
        assigned_centroids = centroids[labels.astype(cp.int32, copy=False)]
        sq_dists = cp.sum(cp.square(feats_cp - assigned_centroids), axis=1)
        selected = _select_min_distance_representatives(labels, sq_dists)
        return _ensure_target_size(selected.tolist(), N, k, rng)
    except Exception as exc:
        if not _KMEANS_RAPIDS_FALLBACK_WARNED:
            warnings.warn(
                f"RAPIDS KMeans unavailable, falling back to non-RAPIDS backend. "
                f"Reason: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
            _KMEANS_RAPIDS_FALLBACK_WARNED = True
        return _subsample_kmeans_fallback(features, patch_ratio, rng)


def _subsample_kmedoid(
    features: torch.Tensor,
    patch_ratio: float,
    rng: np.random.Generator,
    max_iter: int = 3,
) -> np.ndarray:
    """Approximate k-medoids on patch embeddings.

    1. Initialise medoids from the k-means representatives.
    2. Alternate {assign-to-nearest-medoid, pick-cluster-medoid-minimising-sum}.
    3. Stop on convergence or after ``max_iter``.
    """
    N = len(features)
    k = max(1, int(round(N * patch_ratio)))
    if k >= N:
        return np.arange(N)

    feats_src = features if isinstance(features, torch.Tensor) else torch.as_tensor(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = feats_src.to(device=device, dtype=torch.float32).contiguous()

    init_idx = _subsample_kmeans(features, patch_ratio, rng)
    medoid_idx = torch.as_tensor(init_idx, dtype=torch.long, device=device)

    for _ in range(max_iter):
        medoids = feats[medoid_idx]
        cluster_ids = torch.argmin(torch.cdist(feats, medoids, p=2), dim=1)

        updated: List[int] = []
        for c in range(k):
            member_idx = torch.where(cluster_ids == c)[0]
            if member_idx.numel() == 0:
                continue
            member_feats = feats[member_idx]
            intra = torch.cdist(member_feats, member_feats, p=2)
            best_local = torch.argmin(intra.sum(dim=1))
            updated.append(int(member_idx[best_local].item()))

        updated_np = _ensure_target_size(updated, N, k, rng)
        updated_t = torch.as_tensor(updated_np, dtype=torch.long, device=device)
        if torch.equal(torch.sort(updated_t).values, torch.sort(medoid_idx).values):
            medoid_idx = updated_t
            break
        medoid_idx = updated_t

    return np.sort(medoid_idx.detach().cpu().numpy().astype(np.int64, copy=False))


def _subsample_hdbscan(
    features: torch.Tensor,
    patch_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """RAPIDS cuML HDBSCAN — keep one representative per dense cluster.

    HDBSCAN decides the cluster count for us, so we set ``min_cluster_size``
    so the average partition size matches ``N / k``. The largest, most
    confident clusters win the budget; if HDBSCAN finds fewer than ``k``
    clusters we top up with random picks; if it finds none we fall back to
    RAPIDS KMeans.
    """
    cp, _, CuMLHDBSCAN = _import_rapids_cluster_modules()

    N = len(features)
    k = max(1, int(round(N * patch_ratio)))
    if k >= N:
        return np.arange(N)

    _, feats_cp = _to_rapids_feature_matrix(features)
    min_cluster_size = max(2, min(N, int(np.ceil(N / max(k, 1)))))
    min_samples = max(1, min_cluster_size // 2)
    hdb = CuMLHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        output_type="cupy",
    )
    labels = cp.asarray(hdb.fit_predict(feats_cp))
    probs = cp.asarray(getattr(hdb, "probabilities_", cp.ones(N, dtype=cp.float32)))

    cluster_rows = []
    for cluster_id in cp.asnumpy(cp.unique(labels)).tolist():
        if int(cluster_id) < 0:
            continue
        member_indices = cp.where(labels == cluster_id)[0]
        if int(member_indices.size) == 0:
            continue
        member_feats = feats_cp[member_indices]
        centroid = member_feats.mean(axis=0)
        dists = cp.linalg.norm(member_feats - centroid, axis=1)
        rep_pos = int(cp.argmin(dists).item())
        rep_idx = int(member_indices[rep_pos].item())
        cluster_size = int(member_indices.size)
        confidence = float(cp.mean(probs[member_indices]).item())
        cluster_rows.append((rep_idx, cluster_size, confidence, int(cluster_id)))

    if not cluster_rows:
        return _subsample_kmeans(features, patch_ratio, rng)

    cluster_rows.sort(key=lambda row: (-row[1], -row[2], row[3]))
    selected = [rep_idx for rep_idx, _, _, _ in cluster_rows[:k]]
    return _ensure_target_size(selected, N, k, rng)


# ── Slide-encoder embedding extraction ──────────────────────────────────────

def titan_extract_embeddings(
    titan,
    paths: List[Path],
    labels: Dict[str, int],
    coord_dir: Path,
    patch_ratio: float,
    device: str,
    seed: int = 42,
    sampling_mode: str = "random",
    feature_cache: Optional[Dict[str, torch.Tensor]] = None,
    fix_alibi: bool = False,
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    slide_feat_dir: Optional[Path] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
    measure_gpu_time: bool = False,
):
    """Run TITAN on a list of slides at ``patch_ratio`` and return the embeddings.

    Parameters
    ----------
    fix_alibi
        When ``True``, pass *all* coordinates through TITAN so the ALiBi
        positional grid stays identical to the no-drop case; dropped patches
        are zeroed so ``preprocess_features`` marks them as background.
    coord_cache
        Pre-loaded ``{slide_id: (coords, tile_size)}``. Skips disk I/O.
    slide_feat_dir
        When set and ``patch_ratio == 1.0``, cache per-slide embeddings as
        ``{slide_feat_dir}/{cancer_type}/{slide_id}.pt``. Allows reusing one
        global cache across runs / budgets.
    custom_index_cache
        Pre-loaded ``{slide_id: indices np.ndarray}`` from a prior sampler.

    Returns
    -------
    ``(embeddings [M, D], labels [M], n_patches [M], slide_ids [M])`` —
    extended by ``(gpu_seconds, selection_seconds)`` when
    ``measure_gpu_time=True``.
    """
    use_slide_cache = slide_feat_dir is not None and patch_ratio == 1.0
    rng = np.random.default_rng(seed)
    embeddings: List[np.ndarray] = []
    label_list: List[int] = []
    n_patch_list: List[int] = []
    slide_id_list: List[str] = []
    n_total = len(paths)
    n_cached = 0
    gpu_time_ms = 0.0
    use_gpu_timer = bool(measure_gpu_time and str(device).startswith("cuda") and torch.cuda.is_available())
    selection_time_seconds = 0.0

    with torch.inference_mode():
        for i, path in enumerate(paths):
            slide_id = path.stem
            cancer_type = path.parent.name
            label = labels.get(slide_id, -1)
            if label == -1:
                continue

            # Per-slide cache (ratio=1.0 only).
            if use_slide_cache:
                cache_path = slide_feat_dir / cancer_type / f"{slide_id}.pt"
                if cache_path.exists():
                    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
                    embeddings.append(saved["emb"].float().numpy())
                    label_list.append(label)
                    n_patch_list.append(saved.get("n_patches", 0))
                    slide_id_list.append(slide_id)
                    n_cached += 1
                    if (i + 1) % 200 == 0 or (i + 1) == n_total:
                        print(f"  [{i+1}/{n_total}] {len(embeddings)} embedded "
                              f"({n_cached} from cache)", end="\r", flush=True)
                    continue

            # Resolve coords (cache or disk).
            if coord_cache is not None and slide_id in coord_cache:
                coords, tile_size = coord_cache[slide_id]
            else:
                coord_path = find_coord_file(coord_dir, cancer_type, slide_id)
                if coord_path is None:
                    continue
                coords, tile_size = load_coords(coord_path)

            features = (
                feature_cache[slide_id]
                if feature_cache is not None and slide_id in feature_cache
                else load_features(path)
            )

            t0 = time.perf_counter()
            idx = subsample_indices(
                len(features), patch_ratio, rng,
                mode=sampling_mode, coords=coords, features=features,
                custom_indices=None if custom_index_cache is None else custom_index_cache.get(slide_id),
            )
            selection_time_seconds += time.perf_counter() - t0
            n_kept = len(idx)

            if fix_alibi and len(features) == len(coords):
                # Pass the full grid; zero out dropped patches → bg_mask.
                full_features = torch.zeros_like(features)
                full_features[torch.from_numpy(idx)] = features[torch.from_numpy(idx)]
                feat_t = full_features.unsqueeze(0).to(device, dtype=torch.float32)
                coord_t = torch.from_numpy(coords).unsqueeze(0).to(device)
            else:
                feat_t = features[torch.from_numpy(idx)].unsqueeze(0).to(device, dtype=torch.float32)
                coord_t = torch.from_numpy(coords[idx]).unsqueeze(0).to(device)

            if titan is None:
                print(f"\n  [warn] titan=None but {slide_id} not in cache; skipping.")
                continue

            if use_gpu_timer:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            slide_emb = titan.forward_slide(feat_t, coord_t, tile_size)  # [1, D]
            if use_gpu_timer:
                end_event.record()
                torch.cuda.synchronize()
                gpu_time_ms += float(start_event.elapsed_time(end_event))
            emb_cpu = slide_emb.squeeze(0).float().cpu()

            if use_slide_cache:
                cache_path = slide_feat_dir / cancer_type / f"{slide_id}.pt"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"emb": emb_cpu, "n_patches": n_kept}, cache_path)

            embeddings.append(emb_cpu.numpy())
            label_list.append(label)
            n_patch_list.append(n_kept)
            slide_id_list.append(slide_id)

            if (i + 1) % 200 == 0 or (i + 1) == n_total:
                print(f"  [{i+1}/{n_total}] {len(embeddings)} embedded "
                      f"({n_cached} from cache)", end="\r", flush=True)

    print()
    out = (np.stack(embeddings), np.array(label_list), n_patch_list, slide_id_list)
    if use_gpu_timer:
        return out + (gpu_time_ms / 1000.0, selection_time_seconds)
    return out


def gigapath_extract_embeddings(
    gigapath_enc,
    paths: List[Path],
    labels: Dict[str, int],
    coord_dir: Path,
    patch_ratio: float,
    device: str,
    seed: int = 42,
    sampling_mode: str = "random",
    feature_cache: Optional[Dict[str, torch.Tensor]] = None,
    coord_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    slide_feat_dir: Optional[Path] = None,
    feat_layers: Optional[List[int]] = None,
    custom_index_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Run Prov-GigaPath on a list of slides at ``patch_ratio``.

    Same caching / sampling semantics as ``titan_extract_embeddings``. When
    ``feat_layers`` is given, the slide encoder is run with
    ``all_layer_embed=True`` and the requested layer outputs are concatenated
    along the last dimension (used by the ablation that probes intermediate
    LongNet layers).
    """
    use_slide_cache = (slide_feat_dir is not None and patch_ratio == 1.0 and feat_layers is None)
    rng = np.random.default_rng(seed)
    embeddings: List[np.ndarray] = []
    label_list: List[int] = []
    n_patch_list: List[int] = []
    n_total = len(paths)
    n_cached = 0

    with torch.inference_mode():
        for i, path in enumerate(paths):
            slide_id = path.stem
            cancer_type = path.parent.name
            label = labels.get(slide_id, -1)
            if label == -1:
                continue

            if use_slide_cache:
                cache_path = slide_feat_dir / cancer_type / f"{slide_id}.pt"
                if cache_path.exists():
                    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
                    embeddings.append(saved["emb"].float().numpy())
                    label_list.append(label)
                    n_patch_list.append(saved.get("n_patches", 0))
                    n_cached += 1
                    if (i + 1) % 200 == 0 or (i + 1) == n_total:
                        print(f"  [{i+1}/{n_total}] {len(embeddings)} embedded "
                              f"({n_cached} from cache)", end="\r", flush=True)
                    continue

            if coord_cache is not None and slide_id in coord_cache:
                coords, tile_size = coord_cache[slide_id]
            else:
                coord_path = find_coord_file(coord_dir, cancer_type, slide_id)
                if coord_path is None:
                    continue
                coords, tile_size = load_coords(coord_path)

            features = (
                feature_cache[slide_id]
                if feature_cache is not None and slide_id in feature_cache
                else load_features(path)
            )

            idx = subsample_indices(
                len(features), patch_ratio, rng,
                mode=sampling_mode, coords=coords, features=features,
                custom_indices=None if custom_index_cache is None else custom_index_cache.get(slide_id),
            )
            n_kept = len(idx)

            feat_t = features[torch.from_numpy(idx)].unsqueeze(0).to(device, dtype=torch.float32)
            coord_t = torch.from_numpy(coords[idx]).float().unsqueeze(0).to(device)

            if gigapath_enc is None:
                print(f"\n  [warn] gigapath_enc=None but {slide_id} not in cache; skipping.")
                continue

            slide_emb = gigapath_enc.forward_slide(
                feat_t, coord_t, tile_size=tile_size,
                all_layer_embed=feat_layers is not None,
            )
            if feat_layers is not None:
                selected = [slide_emb[layer_idx] for layer_idx in feat_layers]
                slide_emb = torch.cat(selected, dim=-1)
            emb_cpu = slide_emb.squeeze(0).float().cpu()

            if use_slide_cache:
                cache_path = slide_feat_dir / cancer_type / f"{slide_id}.pt"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"emb": emb_cpu, "n_patches": n_kept}, cache_path)

            embeddings.append(emb_cpu.numpy())
            label_list.append(label)
            n_patch_list.append(n_kept)

            if (i + 1) % 200 == 0 or (i + 1) == n_total:
                print(f"  [{i+1}/{n_total}] {len(embeddings)} embedded "
                      f"({n_cached} from cache)", end="\r", flush=True)

    print()
    return np.stack(embeddings), np.array(label_list), n_patch_list
