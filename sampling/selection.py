from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# --- Cycle-15 module-level helpers ---

def _scatter_argmin(values: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Per-label argmin: returns index into `values` of the minimum per label."""
    INF = torch.finfo(values.dtype).max
    best_val = values.new_full((num_labels,), INF)
    best_val.scatter_reduce_(0, labels, values, reduce='amin', include_self=True)
    pos = torch.arange(len(values), device=values.device, dtype=values.dtype)
    pos = pos.masked_fill(values > best_val[labels], INF)
    best_pos = values.new_full((num_labels,), INF)
    best_pos.scatter_reduce_(0, labels, pos, reduce='amin', include_self=True)
    return best_pos.to(torch.int64)


def _scatter_argmax(values: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Per-label argmax: returns index into `values` of the maximum per label."""
    INF = torch.finfo(values.dtype).max
    NEGINF = -INF
    best_val = values.new_full((num_labels,), NEGINF)
    best_val.scatter_reduce_(0, labels, values, reduce='amax', include_self=True)
    pos = torch.arange(len(values), device=values.device, dtype=values.dtype)
    pos = pos.masked_fill(values < best_val[labels], INF)
    best_pos = values.new_full((num_labels,), INF)
    best_pos.scatter_reduce_(0, labels, pos, reduce='amin', include_self=True)
    return best_pos.to(torch.int64)


# Lazily-compiled JIT function for cdist pair.
_JIT_CDIST_PAIR = None
_JIT_CDIST_INIT_DONE = False


def _get_jit_cdist_pair():
    global _JIT_CDIST_PAIR, _JIT_CDIST_INIT_DONE
    if _JIT_CDIST_INIT_DONE:
        return _JIT_CDIST_PAIR
    _JIT_CDIST_INIT_DONE = True
    try:
        @torch.jit.script
        def _fn(sel: torch.Tensor, unsel: torch.Tensor, other: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.cdist(sel, other, p=2), torch.cdist(unsel, other, p=2)
        _JIT_CDIST_PAIR = _fn
    except Exception as e:
        print(f"[jit-script] compile failed: {e}; using eager cdist", flush=True)
        _JIT_CDIST_PAIR = None
    return _JIT_CDIST_PAIR


# =============================================================================
# Core FPS (L2, single-start). All alternative pairwise-distance metrics
# (cosine, hybrid), objective classes (kDPP, Riesz, MMR), multi-start restart,
# exclusion-radius masking, patch-ID blending, magnitude/hubness/entropy biases,
# and margin-aware variants were closed in cycles 22-29 — see CONJECTURE.md.
# =============================================================================


@torch.no_grad()
def sample_cluster_fps(
    samples: torch.Tensor,
    centroid: torch.Tensor,
    available_indices: torch.Tensor,
    seed_indices: torch.Tensor,
    num_pick: int,
    fps_metric: str = "l2",
    fps_l2_normalize: bool = False,
    fps_hybrid_alpha: float = 1.0,
    fps_magnitude_beta: float = 0.0,
    fps_aggregate_mode: str = "min",
) -> torch.Tensor:
    """Greedy farthest-point sampling inside a cluster.

    fps_metric='l2'     — squared L2 distance (legacy default).
    fps_metric='cosine' — cosine distance 1-cos(x,y); feature vectors are
                          normalized only for the distance computation, not
                          modified in-place.
    fps_l2_normalize    — if True with fps_metric='l2', normalize features to
                          unit sphere before L2 distance computation only; raw
                          features are unchanged downstream (decoupled metric).

    If ``seed_indices`` is non-empty, min-distance is initialized against the
    seed set (cluster reps, or the global repulsion set for fps_seed_global*).
    If empty, the first pick is the centroid-nearest available sample.
    """
    if num_pick <= 0 or available_indices.numel() == 0:
        return torch.empty((0,), device=available_indices.device, dtype=torch.int64)

    available_feats = samples.index_select(0, available_indices)
    available_fps_feats = available_feats

    # Magnitude weighting: w_i = (||x_i|| / median_norm)^(beta/2).
    # Applied multiplicatively to min_dists so high-magnitude patches get boosted selection.
    if fps_magnitude_beta > 0.0:
        _avail_norms = available_feats.norm(dim=1).clamp_min_(1e-12)
        _median_norm = _avail_norms.median().clamp_min_(1e-12)
        _avail_w = (_avail_norms / _median_norm).pow(fps_magnitude_beta * 0.5)
    else:
        _avail_w = None

    if fps_metric == "cosine":
        avail_n = F.normalize(available_feats, dim=-1, eps=1e-12)
        if seed_indices.numel() > 0:
            seed_feats = samples.index_select(0, seed_indices)
            seed_n = F.normalize(seed_feats, dim=-1, eps=1e-12)
            min_dists = (1.0 - (avail_n @ seed_n.T)).min(dim=1).values
        else:
            centroid_n = F.normalize(centroid.view(1, -1), dim=-1, eps=1e-12)
            min_dists = (1.0 - (avail_n @ centroid_n.T)).squeeze(1)
    elif fps_l2_normalize and fps_hybrid_alpha < 1.0:
        # Hybrid blend: α*d_L2_norm² + (1−α)*(1−cos). Requires fps_l2_normalize=True.
        avail_n = F.normalize(available_feats, dim=-1, eps=1e-12)
        if seed_indices.numel() > 0:
            seed_feats = samples.index_select(0, seed_indices)
            seed_n = F.normalize(seed_feats, dim=-1, eps=1e-12)
            d_l2sq = torch.cdist(avail_n, seed_n, p=2).square()
            d_cos = 1.0 - (avail_n @ seed_n.T)
            d_blend = fps_hybrid_alpha * d_l2sq + (1.0 - fps_hybrid_alpha) * d_cos
            min_dists = d_blend.min(dim=1).values
        else:
            centroid_n = F.normalize(centroid.view(1, -1), dim=-1, eps=1e-12)
            d_l2sq = (avail_n - centroid_n).square().sum(dim=1)
            d_cos = 1.0 - (avail_n @ centroid_n.T).squeeze(1)
            min_dists = fps_hybrid_alpha * d_l2sq + (1.0 - fps_hybrid_alpha) * d_cos
    elif fps_l2_normalize:
        # Decoupled-metric: normalize for distance only; raw features unchanged downstream.
        avail_n = F.normalize(available_feats, dim=-1, eps=1e-12)
        if seed_indices.numel() > 0:
            seed_feats = samples.index_select(0, seed_indices)
            seed_n = F.normalize(seed_feats, dim=-1, eps=1e-12)
            min_dists = torch.cdist(avail_n, seed_n, p=2).square().min(dim=1).values
        else:
            centroid_n = F.normalize(centroid.view(1, -1), dim=-1, eps=1e-12)
            min_dists = (avail_n - centroid_n).square().sum(dim=1)
    else:
        avail_n = None
        if seed_indices.numel() > 0:
            seed_feats_src = samples.index_select(0, seed_indices)
            _all_dists_sq = torch.cdist(available_fps_feats, seed_feats_src, p=2).square()
            if fps_aggregate_mode == "median":
                # max-median-distance FPS. More robust than max-min when seeds are outlier-heavy.
                min_dists = _all_dists_sq.median(dim=1).values
            else:
                min_dists = _all_dists_sq.min(dim=1).values
        else:
            min_dists = (available_fps_feats - centroid.view(1, -1)).square().sum(dim=1)

    if _avail_w is not None:
        min_dists = min_dists * _avail_w

    chosen_local: list[torch.Tensor] = []
    chosen_mask = torch.zeros(
        available_indices.size(0), device=available_indices.device, dtype=torch.bool
    )

    for _ in range(min(num_pick, int(available_indices.numel()))):
        scores = min_dists.masked_fill(chosen_mask, float("-inf"))
        best_local = scores.argmax()
        if chosen_mask[best_local]:
            break

        chosen_local.append(best_local.view(1))
        chosen_mask[best_local] = True

        if fps_metric == "cosine":
            chosen_n = avail_n[best_local].view(1, -1)  # type: ignore[index]
            new_dist = (1.0 - (avail_n @ chosen_n.T)).squeeze(1)  # type: ignore[operator]
        elif fps_l2_normalize and fps_hybrid_alpha < 1.0:
            chosen_n = avail_n[best_local].view(1, -1)  # type: ignore[index]
            d_l2sq = (avail_n - chosen_n).square().sum(dim=1)  # type: ignore[operator]
            d_cos = 1.0 - (avail_n @ chosen_n.T).squeeze(1)  # type: ignore[operator]
            new_dist = fps_hybrid_alpha * d_l2sq + (1.0 - fps_hybrid_alpha) * d_cos
        elif fps_l2_normalize:
            chosen_n = avail_n[best_local].view(1, -1)  # type: ignore[index]
            new_dist = (avail_n - chosen_n).square().sum(dim=1)  # type: ignore[operator]
        else:
            chosen_fps_feat = available_fps_feats[best_local].view(1, -1)
            new_dist = (available_fps_feats - chosen_fps_feat).square().sum(dim=1)
        if _avail_w is not None:
            new_dist = new_dist * (_avail_w * _avail_w[best_local])

        min_dists = torch.minimum(min_dists, new_dist)

    if not chosen_local:
        return torch.empty((0,), device=available_indices.device, dtype=torch.int64)

    chosen_local_idx = torch.cat(chosen_local, dim=0)
    return available_indices.index_select(0, chosen_local_idx)


# =============================================================================
# Facility-location greedy coreset fill.
# Greedy max-sum-of-coverage-benefit: at each step picks the candidate x that
# maximally reduces sum_y max(0, cur_min(y) - d(y,x)) where cur_min(y) is the
# min distance from y to the current selection.  Same O(N×K) asymptotic as FPS.
# =============================================================================


@torch.no_grad()
def sample_cluster_facility_location(
    samples: torch.Tensor,
    centroid: torch.Tensor,
    available_indices: torch.Tensor,
    seed_indices: torch.Tensor,
    num_pick: int,
) -> torch.Tensor:
    if num_pick <= 0 or available_indices.numel() == 0:
        return torch.empty((0,), device=available_indices.device, dtype=torch.int64)

    available_feats = samples.index_select(0, available_indices)  # (A, D)
    A = int(available_feats.size(0))

    if seed_indices.numel() > 0:
        seed_feats = samples.index_select(0, seed_indices)
        cur_min = torch.cdist(available_feats, seed_feats, p=2).min(dim=1).values  # (A,)
    else:
        cur_min = (available_feats - centroid.view(1, -1)).norm(dim=1)  # (A,)

    # Precompute all-pairs distances once (O(A²D)) — A is small per cluster.
    if A <= 8192:
        d_aa = torch.cdist(available_feats, available_feats, p=2)  # (A, A)
    else:
        d_aa = None  # fall back to block-wise below

    chosen_mask = torch.zeros(A, dtype=torch.bool, device=available_feats.device)
    picks_local: list[int] = []

    for _ in range(min(num_pick, A)):
        if d_aa is not None:
            # benefit[j] = sum_i clamp(cur_min[i] - d[i,j], min=0)
            benefit = torch.clamp(cur_min.unsqueeze(1) - d_aa, min=0.0).sum(dim=0)
        else:
            block = 512
            benefit = torch.zeros(A, device=available_feats.device)
            for start in range(0, A, block):
                end = min(start + block, A)
                d_block = torch.cdist(available_feats, available_feats[start:end], p=2)
                benefit[start:end] = torch.clamp(cur_min.unsqueeze(1) - d_block, min=0.0).sum(dim=0)

        benefit.masked_fill_(chosen_mask, float("-inf"))
        best_local = int(benefit.argmax().item())
        if chosen_mask[best_local]:
            break

        picks_local.append(best_local)
        chosen_mask[best_local] = True
        d_to_new = (available_feats - available_feats[best_local].unsqueeze(0)).norm(dim=1)
        cur_min = torch.minimum(cur_min, d_to_new)

    if not picks_local:
        return torch.empty((0,), device=available_indices.device, dtype=torch.int64)

    picks_tensor = torch.tensor(picks_local, device=available_indices.device, dtype=torch.int64)
    return available_indices.index_select(0, picks_tensor)


# =============================================================================
# Budget allocation — only uniform and sqrt survive cycles 22-27.
# =============================================================================


def allocate_cluster_extra_budget(
    available_counts: np.ndarray,
    total_extra: int,
    mode: str,
    entropy_temperature: float = 1.0,
    cluster_sizes: np.ndarray | None = None,
    cluster_mean_intra_dists: np.ndarray | None = None,
    cluster_within_var: np.ndarray | None = None,
) -> np.ndarray:
    quotas = np.zeros_like(available_counts, dtype=np.int64)
    if total_extra <= 0 or available_counts.sum() <= 0:
        return quotas

    if total_extra > int(available_counts.sum()):
        raise ValueError(
            f"Requested total_extra={total_extra} exceeds available capacity={int(available_counts.sum())}."
        )

    positive = available_counts > 0
    if mode == "uniform":
        weights = positive.astype(np.float64)
    elif mode == "size":
        # Linear-proportional to cluster size (ablation A6).
        sizes = cluster_sizes if cluster_sizes is not None else available_counts
        weights = np.where(positive, sizes.astype(np.float64), 0.0)
    elif mode == "sqrt":
        weights = np.sqrt(available_counts.astype(np.float64))
    elif mode == "entropy":
        # Entropy-weighted allocation: p_c = cluster_size / N, score = -p*log(p).
        # weights = softmax(score / tau). Concentrates budget on mid-size clusters.
        sizes = cluster_sizes if cluster_sizes is not None else available_counts
        sizes_f = sizes.astype(np.float64)
        total_n = float(sizes_f.sum())
        if total_n <= 0:
            weights = positive.astype(np.float64)
        else:
            p = sizes_f / total_n
            score = -p * np.log(np.maximum(p, 1e-12))
            tau = float(entropy_temperature) if entropy_temperature > 0 else 1.0
            score_scaled = score / tau
            score_scaled -= score_scaled.max()  # numerical stability
            exp_s = np.exp(score_scaled)
            weights = exp_s / exp_s.sum()
        weights[~positive] = 0.0
    elif mode == "density":
        # Inverse-density allocation: weights ∝ mean intra-cluster distance.
        # Diffuse clusters (high scatter) receive more budget; tight duplicate-heavy
        # clusters receive less. Orthogonal to size-based (uniform/sqrt) and entropy.
        if cluster_mean_intra_dists is not None:
            dists = cluster_mean_intra_dists.astype(np.float64)
        else:
            dists = available_counts.astype(np.float64)
        weights = np.where(positive, dists, 0.0)
    elif mode == "adaptive_slide_size":
        # Alpha-blended cluster-size power: weight_c ∝ size_c^alpha(N)
        # alpha ∈ [0.5, 1.0], transitions from sqrt (small slides) to linear (large).
        sizes = cluster_sizes if cluster_sizes is not None else available_counts
        N = float(np.sum(sizes))
        scale = 10000.0
        raw_alpha = 0.5 + 0.5 * float(np.tanh(float(np.log1p(N / scale)) - 1.0))
        alpha = float(np.clip(raw_alpha, 0.5, 1.0))
        weights = np.where(positive, np.power(sizes.astype(np.float64).clip(0) + 1e-12, alpha), 0.0)
    elif mode == "within_var":
        # Per-cluster within-cluster feature variance weighted by cluster size.
        # var_k = (1/count_k) * sum_{i: labels[i]==k} ||X[i] - mean_k||^2
        # weight_k = (var_k + eps) * count_k  (size-weighted so small high-var clusters don't monopolize).
        # Fallback to uniform-by-size if all clusters have zero variance.
        if cluster_within_var is not None:
            weights = cluster_within_var.copy()
        else:
            weights = (cluster_sizes if cluster_sizes is not None else available_counts).astype(np.float64)
    elif mode == "within_var_sqrt":
        # Softer weighting: sqrt(var_k + eps) * count_k. Reduces concentration vs
        # linear (var+eps)*count starves background clusters.
        if cluster_within_var is not None:
            weights = cluster_within_var.copy()
        else:
            weights = (cluster_sizes if cluster_sizes is not None else available_counts).astype(np.float64)
    else:
        raise ValueError(f"Unknown budget_allocation mode: {mode!r}.")
    weights[~positive] = 0.0

    if float(weights.sum()) <= 0.0:
        raise RuntimeError("No positive cluster weights available for budget allocation.")

    raw = total_extra * (weights / weights.sum())
    quotas = np.minimum(np.floor(raw).astype(np.int64), available_counts)
    used = int(quotas.sum())
    remaining_capacity = available_counts - quotas
    fractional = raw - np.floor(raw)

    while used < total_extra:
        candidates = np.where(remaining_capacity > 0)[0]
        if candidates.size == 0:
            raise RuntimeError(
                f"Unable to allocate the remaining {total_extra - used} slots across clusters."
            )

        order = sorted(
            candidates.tolist(),
            key=lambda idx: (-fractional[idx], -remaining_capacity[idx], idx),
        )
        progress = False
        for idx in order:
            if used >= total_extra:
                break
            if remaining_capacity[idx] <= 0:
                continue
            quotas[idx] += 1
            remaining_capacity[idx] -= 1
            used += 1
            progress = True
        if not progress:
            raise RuntimeError(
                f"Stalled while allocating remaining budget: total_extra={total_extra}, used={used}."
            )

    return quotas


# =============================================================================
# Per-cluster fill dispatch.
# =============================================================================


def _fps_from_cluster(
    samples: torch.Tensor,
    centroid: torch.Tensor,
    available_indices: torch.Tensor,
    seed_indices: torch.Tensor,
    num_pick: int,
    fps_metric: str = "l2",
    fps_l2_normalize: bool = False,
    fps_hybrid_alpha: float = 1.0,
    fps_magnitude_beta: float = 0.0,
    fps_aggregate_mode: str = "min",
) -> torch.Tensor:
    return sample_cluster_fps(
        samples=samples,
        centroid=centroid,
        available_indices=available_indices,
        seed_indices=seed_indices,
        num_pick=num_pick,
        fps_metric=fps_metric,
        fps_l2_normalize=fps_l2_normalize,
        fps_hybrid_alpha=fps_hybrid_alpha,
        fps_magnitude_beta=fps_magnitude_beta,
        fps_aggregate_mode=fps_aggregate_mode,
    )


@torch.no_grad()
def sample_cluster_additions(
    samples: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    base_indices: torch.Tensor,
    add_sample: str,
    add_sample_num: int,
    fps_metric: str = "l2",
    fps_l2_normalize: bool = False,
    fps_hybrid_alpha: float = 1.0,
    fps_magnitude_beta: float = 0.0,
) -> torch.Tensor:
    """Fixed per-cluster fill used when ``--budget_ratio`` is unset."""
    if add_sample == "none" or add_sample_num <= 0:
        return torch.empty((0,), device=samples.device, dtype=torch.int64)

    num_samples = samples.size(0)
    num_clusters = centroids.size(0)
    device = samples.device

    labels = labels.reshape(-1).to(dtype=torch.int64, device=device)
    if labels.numel() != num_samples:
        raise ValueError(
            f"Label count mismatch: got {labels.numel()} labels for {num_samples} samples."
        )

    counts = torch.bincount(labels, minlength=num_clusters)
    counts_cpu = counts.cpu().tolist()
    order = torch.argsort(labels)

    selected_mask = torch.zeros(num_samples, device=device, dtype=torch.bool)
    if base_indices.numel() > 0:
        selected_mask[base_indices.unique()] = True

    extra_indices: list[torch.Tensor] = []
    offset = 0

    for cluster_id, count in enumerate(counts_cpu):
        if count <= 0:
            continue

        member_indices = order[offset:offset + count]
        offset += count

        member_selected = selected_mask.index_select(0, member_indices)
        seed_indices = member_indices[member_selected]
        available_indices = member_indices[~member_selected]
        if available_indices.numel() == 0:
            continue

        num_pick = min(add_sample_num, int(available_indices.numel()))
        if num_pick <= 0:
            continue

        if add_sample == "fps":
            seeds = seed_indices
        elif add_sample in {"fps_seed_global", "fps_seed_global_cap"}:
            # Fixed-count variant: use a global seed composed of already-selected reps.
            seeds = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        elif add_sample == "facility_location":
            seeds = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        else:
            raise ValueError(f"Unknown add_sample: {add_sample!r}")

        if add_sample == "facility_location":
            chosen = sample_cluster_facility_location(
                samples=samples,
                centroid=centroids[cluster_id],
                available_indices=available_indices,
                seed_indices=seeds,
                num_pick=num_pick,
            )
        else:
            chosen = _fps_from_cluster(
                samples=samples,
                centroid=centroids[cluster_id],
                available_indices=available_indices,
                seed_indices=seeds,
                num_pick=num_pick,
                fps_metric=fps_metric,
                fps_l2_normalize=fps_l2_normalize,
                fps_hybrid_alpha=fps_hybrid_alpha,
                fps_magnitude_beta=fps_magnitude_beta,
            )

        if chosen.numel() == 0:
            continue
        selected_mask[chosen] = True
        extra_indices.append(chosen)

    if not extra_indices:
        return torch.empty((0,), device=device, dtype=torch.int64)
    return torch.cat(extra_indices, dim=0)


@torch.no_grad()
def sample_cluster_additions_to_budget(
    samples: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    base_indices: torch.Tensor,
    target_selected: int,
    add_sample: str,
    budget_allocation: str,
    fps_seed_global_cap: int = 0,
    fps_metric: str = "l2",
    entropy_temperature: float = 1.0,
    fps_l2_normalize: bool = False,
    fps_hybrid_alpha: float = 1.0,
    fps_magnitude_beta: float = 0.0,
    fps_aggregate_mode: str = "min",
) -> torch.Tensor:
    """Exact-budget fill driven by ``--budget_ratio``.

    Allocates quotas across non-empty clusters (uniform, sqrt, or entropy),
    then fills each cluster with FPS seeded by {own reps, global picks, or
    capped HIGH-patch-ID slice of global picks} as per ``add_sample``.
    """
    current_selected = int(base_indices.unique().numel())
    if target_selected <= current_selected:
        return torch.empty((0,), device=samples.device, dtype=torch.int64)
    if add_sample == "none":
        raise ValueError("--budget_ratio requires an add_sample mode other than 'none'.")

    num_samples = samples.size(0)
    num_clusters = centroids.size(0)
    device = samples.device

    labels = labels.reshape(-1).to(dtype=torch.int64, device=device)

    counts = torch.bincount(labels, minlength=num_clusters)
    counts_cpu = counts.cpu().numpy().astype(np.int64, copy=False)
    order = torch.argsort(labels)

    selected_mask = torch.zeros(num_samples, device=device, dtype=torch.bool)
    selected_mask[base_indices.unique()] = True

    available_counts = np.zeros(num_clusters, dtype=np.int64)
    cluster_mean_intra_dists = (
        np.zeros(num_clusters, dtype=np.float64) if budget_allocation == "density" else None
    )

    cluster_within_var: np.ndarray | None = None
    if budget_allocation in ("within_var", "within_var_sqrt"):
        _mean_pc = samples.new_zeros((num_clusters, samples.size(1)))
        _mean_pc.index_add_(0, labels, samples)
        _counts_f = counts.to(samples.dtype).clamp_min_(1.0)
        _mean_pc /= _counts_f.unsqueeze(1)
        _assigned_means = _mean_pc.index_select(0, labels)
        _within_sq = (samples - _assigned_means).square().sum(dim=1)
        _var_pc = samples.new_zeros(num_clusters)
        _var_pc.scatter_add_(0, labels, _within_sq)
        _var_pc = _var_pc / _counts_f
        if budget_allocation == "within_var":
            _weights = (_var_pc + 1e-8) * _counts_f
        else:  # within_var_sqrt
            _weights = (_var_pc + 1e-8).sqrt() * _counts_f
        if float(_weights.sum().item()) <= 1e-8:
            _weights = _counts_f
        cluster_within_var = _weights.cpu().numpy().astype(np.float64, copy=False)

    offset = 0
    for cluster_id, count in enumerate(counts_cpu.tolist()):
        if count <= 0:
            continue
        member_indices = order[offset:offset + count]
        offset += count
        member_selected = selected_mask.index_select(0, member_indices)
        available_counts[cluster_id] = int((~member_selected).sum().item())
        if cluster_mean_intra_dists is not None:
            _mf = samples.index_select(0, member_indices)
            _ctr = centroids[cluster_id].unsqueeze(0)
            cluster_mean_intra_dists[cluster_id] = float(
                (_mf - _ctr).norm(dim=1).mean().item()
            )

    total_extra = target_selected - current_selected
    quotas = allocate_cluster_extra_budget(
        available_counts=available_counts,
        total_extra=total_extra,
        mode=budget_allocation,
        entropy_temperature=entropy_temperature,
        cluster_sizes=counts_cpu,
        cluster_mean_intra_dists=cluster_mean_intra_dists,
        cluster_within_var=cluster_within_var,
    )

    _USES_GLOBAL_SEED = add_sample in {"fps_seed_global", "fps_seed_global_cap", "facility_location"}
    sorted_selected: torch.Tensor | None = (
        torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        if _USES_GLOBAL_SEED
        else None
    )

    extra_indices: list[torch.Tensor] = []
    offset = 0
    for cluster_id, count in enumerate(counts_cpu.tolist()):
        if count <= 0:
            continue

        member_indices = order[offset:offset + count]
        offset += count
        quota = int(quotas[cluster_id])
        if quota <= 0:
            continue

        member_selected = selected_mask.index_select(0, member_indices)
        seed_indices = member_indices[member_selected]
        available_indices = member_indices[~member_selected]
        if available_indices.numel() == 0:
            continue

        num_pick = min(quota, int(available_indices.numel()))

        if add_sample == "fps":
            seeds = seed_indices
        elif add_sample == "fps_seed_global":
            seeds = sorted_selected  # type: ignore[assignment]
        elif add_sample == "fps_seed_global_cap":
            assert sorted_selected is not None
            if fps_seed_global_cap > 0 and sorted_selected.numel() > fps_seed_global_cap:
                seeds = sorted_selected[-fps_seed_global_cap:]  # HIGH-patch-ID slice
            else:
                seeds = sorted_selected
        elif add_sample == "facility_location":
            seeds = sorted_selected if sorted_selected is not None else seed_indices
        elif add_sample == "random":
            seeds = None  # ablation A8: random within-cluster pick, ignores seeds
        else:
            raise ValueError(f"Unknown add_sample: {add_sample!r}")

        if add_sample == "facility_location":
            chosen = sample_cluster_facility_location(
                samples=samples,
                centroid=centroids[cluster_id],
                available_indices=available_indices,
                seed_indices=seeds,
                num_pick=num_pick,
            )
        elif add_sample == "random":
            # Uniform random pick of `num_pick` from available_indices (ablation A8).
            n_avail = int(available_indices.numel())
            perm = torch.randperm(n_avail, device=available_indices.device)
            chosen = available_indices[perm[:num_pick]]
        else:
            chosen = _fps_from_cluster(
                samples=samples,
                centroid=centroids[cluster_id],
                available_indices=available_indices,
                seed_indices=seeds,
                num_pick=num_pick,
                fps_metric=fps_metric,
                fps_l2_normalize=fps_l2_normalize,
                fps_hybrid_alpha=fps_hybrid_alpha,
                fps_magnitude_beta=fps_magnitude_beta,
                fps_aggregate_mode=fps_aggregate_mode,
            )

        if chosen.numel() == 0:
            continue
        selected_mask[chosen] = True
        extra_indices.append(chosen)
        if _USES_GLOBAL_SEED:
            merged = torch.cat([sorted_selected, chosen])  # type: ignore[arg-type]
            sorted_selected, _ = torch.sort(merged)

    if not extra_indices:
        return torch.empty((0,), device=device, dtype=torch.int64)

    extra_idx = torch.cat(extra_indices, dim=0)
    if int(extra_idx.numel()) != total_extra:
        raise RuntimeError(
            f"Budget fill mismatch: expected {total_extra} extra indices, got {int(extra_idx.numel())}."
        )
    return extra_idx


# =============================================================================
# Margin-swap refinement (anti-ms, sign=-1, is the 0.01 leaderboard piece).
# The forward (sign=+1) direction was closed at every tested cell.
# =============================================================================


@torch.no_grad()
def swap_refine_margin_per_cluster(
    samples: torch.Tensor,
    labels: torch.Tensor,
    selected_indices: torch.Tensor,
    n_passes: int,
    objective_sign: int = 1,
    margin_swap_min_cluster_size: int = 2,
    margin_swap_cross_cluster_top_k: int = 0,
    margin_swap_vectorize_global_cdist: bool = False,
    margin_swap_top_m_unsel: int = 0,
    margin_swap_vectorize_mode: str = "A",
    margin_swap_variance_skip_threshold: float = 0.0,
    adaptive_n: bool = False,
    other_sel_subsample: int = 0,
    margin_agg: str = "mean",
    margin_swap_other_sel_approx: str = "none",
    margin_swap_batch_k: int = 1,
    margin_swap_chain_k: int = 1,
    post_antims_max_spread_k: int = 0,
    profile_anti_ms: bool = False,
    margin_swap_vectorize_full_kloop: int = 0,
    margin_swap_jit_script: bool = False,
    margin_swap_agg_weight: str = "uniform",
    margin_swap_chain_k_top_pct: float = 0.0,
    margin_swap_post_fvec_seq_top_pct: float = 0.0,
    margin_swap_post_fvec_seq_mode: str = "single",
    margin_swap_post_fvec_seq_pass2_top_pct: float = 0.0,
    margin_swap_post_fvec_seq_pass2_mode: str = "stale",
    margin_swap_post_fvec_seq_chain_k: int = 2,
    margin_swap_num_candidates: int = 1,
    margin_swap_torchcompile_mode: str = "none",
    margin_swap_fvec_microbatch_b: int = 0,
    margin_swap_inter_cluster_spread_weight: float = 0.0,
    margin_swap_adaptive_np_large_threshold: int = 0,
    margin_swap_post_fvec_seq_select_mode: str = "delta_rank",
) -> torch.Tensor:
    """Per-cluster mean-margin swap refinement.

    For each cluster c, margin(x) = mean dist to other-cluster selected
    - mean dist to same-cluster unselected. Per-cluster best swap per pass;
    early exit when a pass has no improvement. Preserves |S| and cluster
    membership. objective_sign=+1 maximizes the sum (closed). -1 minimizes
    

    margin_swap_vectorize_global_cdist=True: precompute d(all_sel, all_sel)
    and d(samples, all_sel) once and use index_select lookups for per-cluster
    d_sel_other and d_unsel_other. Two modes:
      Strategy A (default): recompute after each swap (semantically equivalent).
      Strategy B: recompute only at pass-boundary (approximate; fewer rebuilds).
    margin_swap_top_m_unsel>0: restrict unsel pool per cluster to the M nearest
    patches to the cluster centroid, reducing cdist size.
    """
    if (n_passes <= 0 and post_antims_max_spread_k <= 0) or selected_indices.numel() == 0:
        return selected_indices

    device = samples.device
    labels_flat = labels.reshape(-1).to(dtype=torch.int64, device=device)
    num_clusters = int(labels_flat.max().item()) + 1

    sel_unique = selected_indices.unique()
    if int(sel_unique.numel()) != int(selected_indices.numel()):
        return selected_indices

    selected_mask = torch.zeros(samples.size(0), dtype=torch.bool, device=device)
    selected_mask[sel_unique] = True

    # Pre-compute cluster centroids for cross-cluster candidate pool.
    if margin_swap_cross_cluster_top_k > 0:
        _D = samples.size(1)
        cluster_centroids = torch.zeros(num_clusters, _D, device=device)
        cluster_centroids.index_add_(0, labels_flat, samples)
        _cc = torch.bincount(labels_flat, minlength=num_clusters).float().clamp_min_(1.0)
        cluster_centroids = cluster_centroids / _cc.unsqueeze(1)
    else:
        cluster_centroids = None

    def _build_global_precompute():
        asg = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        asf = samples.index_select(0, asg)
        dss = torch.cdist(asf, asf, p=2)                      # (S, S)
        dns = torch.cdist(samples, asf, p=2)                   # (N, S)
        sp = torch.full((samples.size(0),), -1, device=device, dtype=torch.int64)
        sp[asg] = torch.arange(asg.numel(), device=device)
        sc = labels_flat[asg]                                  # cluster label per sel pos
        return dss, dns, sp, sc

    if margin_swap_vectorize_global_cdist:
        g_dss, g_dns, g_sp, g_sc = _build_global_precompute()

    # Variance-aware cluster skip: precompute per-cluster mean
    # intra-distance vectorized (index_add on patch-to-centroid L2 norms). Tight
    # clusters (low intra-dist) rarely have feasible improving swaps; skipping
    # them reduces Python K-loop iterations without rebuilding cdists.
    if margin_swap_variance_skip_threshold > 0.0:
        _D = samples.size(1)
        _counts = torch.bincount(labels_flat, minlength=num_clusters).float().clamp_min_(1.0)
        _csum = torch.zeros(num_clusters, _D, device=device)
        _csum.index_add_(0, labels_flat, samples)
        _centroids_all = _csum / _counts.unsqueeze(1)                    # (K, D)
        _patch_intra = (samples - _centroids_all[labels_flat]).norm(dim=1)  # (N,)
        _intra_sum = torch.zeros(num_clusters, device=device)
        _intra_sum.index_add_(0, labels_flat, _patch_intra)
        _cluster_mean_intra = _intra_sum / _counts                       # (K,)
        _skip_threshold_val = float(
            margin_swap_variance_skip_threshold * _cluster_mean_intra.median().item()
        )
    else:
        _cluster_mean_intra = None
        _skip_threshold_val = 0.0

    def _agg(t: torch.Tensor) -> torch.Tensor:
        if margin_agg == "median":
            return t.median(dim=1).values
        return t.mean(dim=1)

    # --- Full K-loop vectorized path ---
    _ran_fvec_passes = False
    _fvec_cluster_deltas: torch.Tensor | None = None  # accumulated per-cluster fvec deltas (c16)
    _spread_weight = float(margin_swap_inter_cluster_spread_weight)
    _adaptive_np_threshold = int(margin_swap_adaptive_np_large_threshold)
    if n_passes > 0 and margin_swap_vectorize_full_kloop:
        _ran_fvec_passes = True
        _D = samples.size(1)
        _fvec_active = torch.ones(num_clusters, dtype=torch.bool, device=device)
        # Track per-cluster fvec deltas for post-fvec sequential residual
        if margin_swap_post_fvec_seq_top_pct > 0.0:
            _fvec_cluster_deltas = torch.zeros(num_clusters, device=device)

        def _run_one_fvec_step(active_filter):
            _sel_idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
            _unsel_idx = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            _O = int(_sel_idx.numel())
            _U = int(_unsel_idx.numel())
            if _O == 0 or _U == 0:
                return False, None
            _sf = samples.index_select(0, _sel_idx)
            _uf = samples.index_select(0, _unsel_idx)
            _sl = labels_flat[_sel_idx]
            _ul = labels_flat[_unsel_idx]
            _dss = torch.cdist(_sf, _sf, p=2)
            _dus = torch.cdist(_uf, _sf, p=2)
            _same_ss = _sl.unsqueeze(0) == _sl.unsqueeze(1)
            _same_ss_nd = _same_ss & ~torch.eye(_O, dtype=torch.bool, device=device)
            _other_ss = ~_same_ss
            _same_us = _ul.unsqueeze(1) == _sl.unsqueeze(0)
            _other_us = ~_same_us
            if margin_swap_agg_weight == "margin_magnitude":
                # Two-pass: uniform weights first, then abs(margin) as weights
                _w_u = torch.ones(_O, device=device, dtype=_sf.dtype)
                _w_row_u = _w_u.unsqueeze(0)
                _ow_u = _other_ss.float() * _w_row_u
                _ow_s_u = _ow_u.sum(dim=1).clamp_min_(1e-8)
                _m_sel_oth_u = (_dss * _ow_u).sum(dim=1) / _ow_s_u
                _sw_u = _same_ss_nd.float() * _w_row_u
                _sw_s_u = _sw_u.sum(dim=1).clamp_min_(1e-8)
                _m_sel_sam_u = (_dss * _sw_u).sum(dim=1) / _sw_s_u
                _mg_prelim = _m_sel_oth_u - _m_sel_sam_u
                _w_j = _mg_prelim.abs().clamp_min_(1e-6)
            elif margin_swap_agg_weight == "cluster_size":
                _cls_sz = torch.bincount(labels_flat, minlength=num_clusters)
                _w_j = 1.0 / _cls_sz[_sl].float().clamp_min_(1.0).sqrt()
            elif margin_swap_agg_weight == "uncertainty":
                _csum = torch.zeros(num_clusters, _D, device=device)
                _csum.index_add_(0, labels_flat, samples)
                _ccnt = torch.bincount(labels_flat, minlength=num_clusters).float().clamp_min_(1.0)
                _cents = _csum / _ccnt.unsqueeze(1)
                _cent_j = _cents[_sl]
                _d_own = (_sf - _cent_j).norm(dim=1)
                _w_j = 1.0 / _d_own.clamp_min_(1e-4)
            else:
                _w_j = torch.ones(_O, device=device, dtype=_sf.dtype)
            _w_row = _w_j.unsqueeze(0)
            _ow = _other_ss.float() * _w_row
            _ow_s = _ow.sum(dim=1).clamp_min_(1e-8)
            _m_sel_oth = (_dss * _ow).sum(dim=1) / _ow_s
            _sw = _same_ss_nd.float() * _w_row
            _sw_s = _sw.sum(dim=1).clamp_min_(1e-8)
            _m_sel_sam = (_dss * _sw).sum(dim=1) / _sw_s
            _ouw = _other_us.float() * _w_row
            _ouw_s = _ouw.sum(dim=1).clamp_min_(1e-8)
            _m_unsel_oth = (_dus * _ouw).sum(dim=1) / _ouw_s
            _suw = _same_us.float() * _w_row
            _suw_s = _suw.sum(dim=1).clamp_min_(1e-8)
            _m_unsel_sam = (_dus * _suw).sum(dim=1) / _suw_s
            _mg_sel = _m_sel_oth - _m_sel_sam
            _mg_unsel = _m_unsel_oth - _m_unsel_sam
            _ems = _mg_sel * float(objective_sign)
            _emu = _mg_unsel * float(objective_sign)
            # Inter-cluster-spread reward: reward selections far from other-cluster picks.
            if _spread_weight > 0.0 and _O > 1:
                _dss_other = _dss.masked_fill(~_other_ss, float("inf"))
                _min_dist_sel = _dss_other.min(dim=1).values.clamp_min_(0.0)
                _ems = _ems + _spread_weight * _min_dist_sel  # harder to remove well-spread selected
                _dus_other = _dus.masked_fill(~_other_us, float("inf"))
                _min_dist_unsel = _dus_other.min(dim=1).values.clamp_min_(0.0)
                _emu = _emu + _spread_weight * _min_dist_unsel  # easier to add well-spread unselected
            _csz = torch.bincount(labels_flat, minlength=num_clusters)
            _csc = torch.bincount(_sl, minlength=num_clusters)
            _cuc = torch.bincount(_ul, minlength=num_clusters)
            _vc = (_csz >= margin_swap_min_cluster_size) & (_csc > 0) & (_cuc > 0)
            if active_filter is not None:
                _vc = _vc & active_filter
            _INF = torch.finfo(_ems.dtype).max
            _sv = _vc[_sl]
            _uv = _vc[_ul]
            _ems_m = _ems.masked_fill(~_sv, _INF)
            _emu_m = _emu.masked_fill(~_uv, -_INF)
            _bso = _scatter_argmin(_ems_m, _sl, num_clusters)
            _bui = _scatter_argmax(_emu_m, _ul, num_clusters)
            _mo = _O - 1
            _mu = _U - 1
            _bso_c = _bso.clamp(max=_mo)
            _bui_c = _bui.clamp(max=_mu)
            _delta = _emu[_bui_c] - _ems[_bso_c]
            _sa = _vc & (_bso <= _mo) & (_bui <= _mu) & (_delta > 1e-9)
            if not bool(_sa.any().item()):
                return False, None
            _ac = torch.where(_sa)[0]
            selected_mask[_sel_idx[_bso_c[_ac]]] = False
            selected_mask[_unsel_idx[_bui_c[_ac]]] = True
            # Accumulate per-cluster fvec deltas for post-fvec sequential residual (c16)
            if _fvec_cluster_deltas is not None:
                _fvec_cluster_deltas.add_(_delta.abs() * _sa.float())
            return True, _sa

        # torch.compile mode='reduce-overhead' wrapper
        if margin_swap_torchcompile_mode == "reduce-overhead":
            try:
                _fvec_step_fn = torch.compile(_run_one_fvec_step, mode="reduce-overhead", dynamic=True)
                print("[compile-ro] torch.compile reduce-overhead active", flush=True)
            except Exception as _ce:
                print(f"[compile-ro] torch.compile setup failed: {_ce}; using eager", flush=True)
                _fvec_step_fn = _run_one_fvec_step
        else:
            _fvec_step_fn = _run_one_fvec_step

        # Micro-batched fvec: split each pass into B chunks with state updates.
        _mb_B = int(margin_swap_fvec_microbatch_b)
        _mb_active = _mb_B >= 2

        for _fvec_pass in range(n_passes):
            _af = _fvec_active if (adaptive_n and _fvec_pass > 0) else None
            if _mb_active:
                # Build B random-permuted chunks over the K cluster ids.
                _g = torch.Generator(device=device).manual_seed(0xFECB + _fvec_pass)
                _perm = torch.randperm(num_clusters, generator=_g, device=device)
                _chunks = torch.chunk(_perm, _mb_B)
                _any = False
                _new_active_agg = torch.zeros(num_clusters, dtype=torch.bool, device=device) if adaptive_n else None
                for _ci, _chunk_idx in enumerate(_chunks):
                    if _chunk_idx.numel() == 0:
                        continue
                    _chunk_filter = torch.zeros(num_clusters, dtype=torch.bool, device=device)
                    _chunk_filter[_chunk_idx] = True
                    if _af is not None:
                        _chunk_filter = _chunk_filter & _af
                    _any_sub, _new_active_sub = _fvec_step_fn(_chunk_filter)
                    _any = _any or _any_sub
                    if adaptive_n and _new_active_sub is not None:
                        _new_active_agg = _new_active_agg | _new_active_sub
                if adaptive_n:
                    _fvec_active = _new_active_agg if _new_active_agg is not None else torch.zeros(
                        num_clusters, dtype=torch.bool, device=device)
                if margin_swap_chain_k >= 2 and _any:
                    # Run a second chunked sweep for chain-k=2 (still mini-batched).
                    _g2 = torch.Generator(device=device).manual_seed(0xFECB + _fvec_pass + 0xFFFF)
                    _perm2 = torch.randperm(num_clusters, generator=_g2, device=device)
                    _chunks2 = torch.chunk(_perm2, _mb_B)
                    _any2 = False
                    for _chunk_idx in _chunks2:
                        if _chunk_idx.numel() == 0:
                            continue
                        _chunk_filter = torch.zeros(num_clusters, dtype=torch.bool, device=device)
                        _chunk_filter[_chunk_idx] = True
                        if adaptive_n and _fvec_active is not None:
                            _chunk_filter = _chunk_filter & _fvec_active
                        _any_sub2, _ = _fvec_step_fn(_chunk_filter)
                        _any2 = _any2 or _any_sub2
                    _any = _any or _any2
                if not _any:
                    break
            else:
                _any, _new_active = _fvec_step_fn(_af)
                if adaptive_n:
                    _fvec_active = _new_active if _new_active is not None else torch.zeros(
                        num_clusters, dtype=torch.bool, device=device)
                if margin_swap_chain_k >= 2 and _any:
                    _af2 = _new_active if (adaptive_n and _new_active is not None) else None
                    _any2, _new_active2 = _fvec_step_fn(_af2)
                    if adaptive_n and _new_active2 is not None:
                        _fvec_active = _new_active2
                    _any = _any or _any2
                if not _any:
                    break

    # Adaptive np_k extra pass for large clusters: after main fvec passes, run
    # one additional fvec pass restricted to clusters with count >= threshold.
    if (_adaptive_np_threshold > 0 and n_passes > 0 and margin_swap_vectorize_full_kloop
            and _ran_fvec_passes):
        _csz_full = torch.bincount(labels_flat, minlength=num_clusters)
        _large_filter = _csz_full >= _adaptive_np_threshold
        if bool(_large_filter.any().item()):
            _fvec_step_fn(_large_filter)

    # Post-fvec sequential residual setup
    # After fvec converges, run ONE sequential pass on top-pct delta clusters.
    _pfseq_active: torch.Tensor | None = None   # bool (K,) filter or None
    _pfseq_override_chain_k: int | None = None  # overrides margin_swap_chain_k for residual pass
    _pfseq_n_seq_override: int | None = None    # overrides n_passes for sequential loop
    if (margin_swap_vectorize_full_kloop and margin_swap_post_fvec_seq_top_pct > 0.0
            and _ran_fvec_passes and _fvec_cluster_deltas is not None
            and bool(_fvec_cluster_deltas.any().item())):
        _n_top = max(1, int(math.ceil(num_clusters * margin_swap_post_fvec_seq_top_pct)))
        _n_top = min(_n_top, num_clusters)
        if margin_swap_post_fvec_seq_select_mode == "inter_medoid_proximity":
            # Select top-pct clusters by smallest min inter-medoid distance (most contested).
            _sel_idx_imp = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
            _sel_labels_imp = labels_flat[_sel_idx_imp]
            _D_imp = samples.size(1)
            _per_cl_sum = torch.zeros(num_clusters, _D_imp, device=device)
            _per_cl_cnt = torch.zeros(num_clusters, device=device)
            _per_cl_sum.index_add_(0, _sel_labels_imp, samples.index_select(0, _sel_idx_imp))
            _per_cl_cnt.index_add_(0, _sel_labels_imp, torch.ones(_sel_idx_imp.numel(), device=device))
            _per_cl_feat = _per_cl_sum / _per_cl_cnt.clamp_min_(1.0).unsqueeze(1)  # (K, D)
            _imp_dist = torch.cdist(_per_cl_feat, _per_cl_feat, p=2)
            _imp_dist.fill_diagonal_(float("inf"))
            _min_imp_dist = _imp_dist.min(dim=1).values  # (K,) smaller = more contested
            _topk_imp = _min_imp_dist.topk(_n_top, largest=False, sorted=False)
            _pfseq_active = torch.zeros(num_clusters, dtype=torch.bool, device=device)
            _pfseq_active[_topk_imp.indices] = True
        else:
            _topk_deltas = _fvec_cluster_deltas.topk(_n_top, largest=True, sorted=False)
            _pfseq_active = torch.zeros(num_clusters, dtype=torch.bool, device=device)
            _pfseq_active[_topk_deltas.indices] = True
        _pfseq_override_chain_k = (
            int(margin_swap_post_fvec_seq_chain_k) if margin_swap_post_fvec_seq_mode == "chain_k" else 1
        )
        _pfseq_n_seq_override = 1
        _ran_fvec_passes = False  # allow sequential loop to run once on top-pct clusters

    if adaptive_n:
        cluster_active = torch.ones(num_clusters, dtype=torch.bool, device=device)

    # Profile: accumulate per-op timing across all passes and clusters.
    if profile_anti_ms:
        import time as _time
        _prof_totals: dict[str, float] = {
            "cdist_sel_other": 0.0, "cdist_unsel_other": 0.0,
            "cdist_sel_sel": 0.0, "cdist_sel_unsel": 0.0, "cdist_unsel_unsel": 0.0,
            "agg_other": 0.0, "swap_search": 0.0,
        }

    # Iterative pfseq outer loop: runs once (default) or twice (pass2_top_pct > 0 and active).
    _pfseq_outer_iters = (2 if (margin_swap_post_fvec_seq_pass2_top_pct > 0.0
                                and _pfseq_active is not None
                                and _fvec_cluster_deltas is not None) else 1)
    for _pfseq_outer in range(_pfseq_outer_iters):
        if _pfseq_outer == 1:
            if margin_swap_post_fvec_seq_pass2_mode == "recomputed":
                # Reset fvec delta accumulator and run ONE more fvec step on current (post-pass-1) state.
                _fvec_cluster_deltas.zero_()
                _fvec_step_fn(None)
                _n_top2 = max(1, int(math.ceil(num_clusters * margin_swap_post_fvec_seq_pass2_top_pct)))
                _n_top2 = min(_n_top2, num_clusters)
                _topk_p2 = _fvec_cluster_deltas.topk(_n_top2, largest=True, sorted=False)
                _pfseq_active = torch.zeros(num_clusters, dtype=torch.bool, device=device)
                _pfseq_active[_topk_p2.indices] = True
            else:  # stale (c18 C default, unchanged)
                _n_top1 = max(1, int(math.ceil(num_clusters * margin_swap_post_fvec_seq_top_pct)))
                _n_top1 = min(_n_top1, num_clusters)
                _n_top2 = max(1, int(math.ceil(num_clusters * margin_swap_post_fvec_seq_pass2_top_pct)))
                _n_top2 = min(_n_top2, max(1, num_clusters - _n_top1))
                _n_need = min(_n_top1 + _n_top2, num_clusters)
                if _n_top2 <= 0 or _n_need <= _n_top1:
                    break
                _topk_p2 = _fvec_cluster_deltas.topk(_n_need, largest=True, sorted=True)
                _pfseq_active = torch.zeros(num_clusters, dtype=torch.bool, device=device)
                _pfseq_active[_topk_p2.indices[_n_top1:_n_need]] = True
            _ran_fvec_passes = False
        _n_seq_passes = _pfseq_n_seq_override if _pfseq_n_seq_override is not None else n_passes
        for _pass in range(_n_seq_passes):
            if _ran_fvec_passes:
                break
            any_improvement = False
            _needs_global_refresh = False  # Strategy B: track whether a swap happened this pass

            if adaptive_n:
                cluster_changed_this_pass = torch.zeros(num_clusters, dtype=torch.bool, device=device)

            # Per-pass subsample of the other-cluster selected pool (non-vectorized path only).
            # All clusters in this pass see the same subsample for cache coherence.
            _pass_sel_pool: torch.Tensor | None = None
            if other_sel_subsample > 0 and not margin_swap_vectorize_global_cdist:
                _all_sel = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
                if _all_sel.numel() > other_sel_subsample:
                    _g = torch.Generator(device=device).manual_seed(0xC13 + _pass)
                    _perm = torch.randperm(_all_sel.numel(), generator=_g, device=device)
                    _pass_sel_pool = _all_sel[_perm[:other_sel_subsample]]

            # Per-pass cluster-centroid precompute for centroid approximation mode (2a).
            if margin_swap_other_sel_approx == "centroid" and not margin_swap_vectorize_global_cdist:
                _sel_feats_all = samples[selected_mask]
                _sel_labels_all = labels_flat[selected_mask]
                _cent_count = torch.zeros(num_clusters, device=device, dtype=samples.dtype)
                _cent_count.index_add_(0, _sel_labels_all, torch.ones(_sel_labels_all.numel(), device=device, dtype=samples.dtype))
                _cent_sum = torch.zeros(num_clusters, samples.size(1), device=device, dtype=samples.dtype)
                _cent_sum.index_add_(0, _sel_labels_all, _sel_feats_all)
                _cluster_sel_centroid = _cent_sum / _cent_count.clamp_min(1.0).unsqueeze(1)  # (K, D)
            else:
                _cluster_sel_centroid = None
                _cent_count = None

            # Per-pass aggregator weights for other-sel pool.
            _pass_agg_weights: torch.Tensor | None = None
            if margin_swap_agg_weight != "uniform":
                _N = samples.size(0)
                _D_w = samples.size(1)
                if margin_swap_agg_weight == "cluster_size":
                    _cls_sz_w = torch.bincount(labels_flat, minlength=num_clusters)
                    _w_all = 1.0 / _cls_sz_w[labels_flat].float().clamp_min_(1.0).sqrt()
                else:  # uncertainty
                    _csum_w = torch.zeros(num_clusters, _D_w, device=device)
                    _csum_w.index_add_(0, labels_flat, samples)
                    _ccnt_w = torch.bincount(labels_flat, minlength=num_clusters).float().clamp_min_(1.0)
                    _cents_w = _csum_w / _ccnt_w.unsqueeze(1)
                    _d_own_w = (samples - _cents_w[labels_flat]).norm(dim=1)
                    _w_all = 1.0 / _d_own_w.clamp_min_(1e-4)
                _pass_agg_weights = _w_all  # shape (N,)

            # Chain-k top-pct tracking: track pass-0 deltas, gate chain-k in pass>0.
            _pass1_cluster_deltas: torch.Tensor | None = None
            _chain_eligible_set: torch.Tensor | None = None  # bool (K,) after pass 0
            if margin_swap_chain_k_top_pct > 0.0 and margin_swap_chain_k >= 2:
                _pass1_cluster_deltas = torch.zeros(num_clusters, device=device)

            # Effective chain_k for this pass (post-fvec residual overrides margin_swap_chain_k)
            _eff_chain_k = _pfseq_override_chain_k if _pfseq_override_chain_k is not None else margin_swap_chain_k

            for c in range(num_clusters):
                if adaptive_n and _pass > 0 and not bool(cluster_active[c].item()):
                    continue  # cluster did not swap in previous pass — skip
                # Post-fvec sequential residual filter: skip clusters not in active set (c16)
                if _pfseq_active is not None and not bool(_pfseq_active[c].item()):
                    continue
                if _skip_threshold_val > 0.0 and _cluster_mean_intra is not None and float(_cluster_mean_intra[c].item()) < _skip_threshold_val:
                    continue  # tight cluster — low swap potential, skip
                member_mask = labels_flat == c
                member_idx = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
                if member_idx.numel() < margin_swap_min_cluster_size:
                    continue
                cluster_sel_mask = selected_mask[member_idx]
                if not cluster_sel_mask.any() or cluster_sel_mask.all():
                    continue

                sel_local = torch.nonzero(cluster_sel_mask, as_tuple=False).squeeze(-1)
                unsel_local = torch.nonzero(~cluster_sel_mask, as_tuple=False).squeeze(-1)

                member_feats = samples.index_select(0, member_idx)
                sel_feats = member_feats.index_select(0, sel_local)
                unsel_feats = member_feats.index_select(0, unsel_local)

                # augment unsel pool with nearest cross-cluster unselected patches.
                if margin_swap_cross_cluster_top_k > 0 and num_clusters > 1 and cluster_centroids is not None:
                    _other_unsel_mask = ~selected_mask & ~member_mask
                    _other_unsel_global = torch.nonzero(_other_unsel_mask, as_tuple=False).squeeze(-1)
                    if _other_unsel_global.numel() > 0:
                        _top_k = min(margin_swap_cross_cluster_top_k, int(_other_unsel_global.numel()))
                        _ou_feats = samples.index_select(0, _other_unsel_global)
                        _, _ki = torch.cdist(
                            cluster_centroids[c:c + 1], _ou_feats, p=2
                        ).squeeze(0).topk(_top_k, largest=False)
                        _cross_g = _other_unsel_global.index_select(0, _ki)
                        unsel_feats = torch.cat([unsel_feats, _ou_feats.index_select(0, _ki)], dim=0)
                        unsel_global_for_swap = torch.cat(
                            [member_idx.index_select(0, unsel_local), _cross_g], dim=0
                        )
                    else:
                        unsel_global_for_swap = member_idx.index_select(0, unsel_local)
                else:
                    unsel_global_for_swap = member_idx.index_select(0, unsel_local)

                # truncate unsel pool to top-M nearest to cluster centroid.
                if margin_swap_top_m_unsel > 0 and unsel_feats.size(0) > margin_swap_top_m_unsel:
                    centroid_c = member_feats.mean(dim=0, keepdim=True)
                    d_to_centroid = (unsel_feats - centroid_c).pow(2).sum(dim=1)
                    top_m = min(margin_swap_top_m_unsel, int(d_to_centroid.numel()))
                    keep_local = d_to_centroid.topk(top_m, largest=False).indices.sort().values
                    unsel_feats = unsel_feats.index_select(0, keep_local)
                    unsel_global_for_swap = unsel_global_for_swap.index_select(0, keep_local)

                _centroid_mode = False
                if margin_swap_vectorize_global_cdist:
                    other_sel_pos = torch.nonzero(g_sc != c, as_tuple=False).squeeze(-1)
                    if other_sel_pos.numel() == 0:
                        continue
                    sel_pos_c = g_sp.index_select(0, member_idx.index_select(0, sel_local))
                    d_sel_other = g_dss.index_select(0, sel_pos_c).index_select(1, other_sel_pos)
                    d_unsel_other = g_dns.index_select(0, unsel_global_for_swap).index_select(1, other_sel_pos)
                elif margin_swap_other_sel_approx == "centroid" and _cluster_sel_centroid is not None:
                    # Centroid approximation (2a): replace O-wide other_sel with K-1 cluster centroids.
                    _c_idx = torch.arange(num_clusters, device=device) != c
                    _cent_k = _cluster_sel_centroid[_c_idx]             # (K-1, D)
                    _w_k = _cent_count[_c_idx]                          # (K-1,)
                    _total_w = _w_k.sum().clamp_min_(1.0)
                    if float(_total_w.item()) == 0.0:
                        continue
                    if profile_anti_ms:
                        torch.cuda.synchronize(device); _t0 = _time.perf_counter()
                    _d_sel_c = torch.cdist(sel_feats, _cent_k, p=2)     # (n_sel, K-1)
                    _d_unsel_c = torch.cdist(unsel_feats, _cent_k, p=2) # (n_unsel, K-1)
                    if profile_anti_ms:
                        torch.cuda.synchronize(device)
                        _prof_totals["cdist_sel_other"] += _time.perf_counter() - _t0
                    mean_d_sel_to_other = (_d_sel_c * _w_k.unsqueeze(0)).sum(dim=1) / _total_w
                    mean_d_unsel_to_other = (_d_unsel_c * _w_k.unsqueeze(0)).sum(dim=1) / _total_w
                    _centroid_mode = True
                    # Store _cent_k/_w_k/_total_w for potential chain-k reuse
                    _chain_cent_k = _cent_k
                    _chain_w_k = _w_k
                    _chain_total_w = _total_w
                else:
                    if _pass_sel_pool is not None:
                        # Use per-pass subsampled pool, filtered to exclude cluster c's members.
                        _in_c = member_mask[_pass_sel_pool]
                        other_sel_global = _pass_sel_pool[~_in_c]
                    else:
                        other_sel_global = torch.nonzero(selected_mask & ~member_mask, as_tuple=False).squeeze(-1)
                    if other_sel_global.numel() == 0:
                        continue
                    other_sel_feats = samples.index_select(0, other_sel_global)  # (O, D)
                    if profile_anti_ms:
                        torch.cuda.synchronize(device); _t0 = _time.perf_counter()
                    if margin_swap_batch_k > 1:
                        # Simplified batched-K (2b): combine sel+unsel into one cdist call.
                        _combined = torch.cat([sel_feats, unsel_feats], dim=0)
                        _d_combined = torch.cdist(_combined, other_sel_feats, p=2)
                        d_sel_other = _d_combined[:sel_feats.size(0)]
                        d_unsel_other = _d_combined[sel_feats.size(0):]
                    else:
                        if margin_swap_jit_script:
                            _jit_fn = _get_jit_cdist_pair()
                            if _jit_fn is not None:
                                d_sel_other, d_unsel_other = _jit_fn(sel_feats, unsel_feats, other_sel_feats)
                            else:
                                d_sel_other = torch.cdist(sel_feats, other_sel_feats, p=2)
                                d_unsel_other = torch.cdist(unsel_feats, other_sel_feats, p=2)
                        else:
                            d_sel_other = torch.cdist(sel_feats, other_sel_feats, p=2)
                            d_unsel_other = torch.cdist(unsel_feats, other_sel_feats, p=2)
                    if profile_anti_ms:
                        torch.cuda.synchronize(device)
                        _prof_totals["cdist_sel_other"] += _time.perf_counter() - _t0

                if not _centroid_mode:
                    if profile_anti_ms:
                        torch.cuda.synchronize(device); _t0 = _time.perf_counter()
                    if _pass_agg_weights is not None:
                        _ow = _pass_agg_weights[other_sel_global]  # (O,)
                        _ow_sum = _ow.sum().clamp_min_(1e-8)
                        mean_d_sel_to_other = (d_sel_other * _ow.unsqueeze(0)).sum(dim=1) / _ow_sum
                        mean_d_unsel_to_other = (d_unsel_other * _ow.unsqueeze(0)).sum(dim=1) / _ow_sum
                    else:
                        mean_d_sel_to_other = _agg(d_sel_other)
                        mean_d_unsel_to_other = _agg(d_unsel_other)
                    if profile_anti_ms:
                        torch.cuda.synchronize(device)
                        _prof_totals["agg_other"] += _time.perf_counter() - _t0

                if profile_anti_ms:
                    torch.cuda.synchronize(device); _t0 = _time.perf_counter()
                d_sel_sel = torch.cdist(sel_feats, sel_feats, p=2)
                d_sel_unsel = torch.cdist(sel_feats, unsel_feats, p=2)
                d_unsel_sel = d_sel_unsel.t()
                d_unsel_unsel = torch.cdist(unsel_feats, unsel_feats, p=2)
                if profile_anti_ms:
                    torch.cuda.synchronize(device)
                    _prof_totals["cdist_sel_sel"] += _time.perf_counter() - _t0

                Kunsel = int(unsel_feats.size(0))  # updated for extended pool
                if Kunsel == 0:
                    continue
                mean_d_sel_to_own_unsel = _agg(d_sel_unsel)
                current_margin = mean_d_sel_to_other - mean_d_sel_to_own_unsel
                current_sum = float(current_margin.sum().item())

                best = None
                Ksel = int(sel_local.numel())

                if Kunsel >= 2:
                    if margin_agg == "median":
                        # Approximation: median of full row (self-dist=0 is minor bias at Kunsel>>1).
                        mean_d_unsel_to_own_unsel_excl_self = d_unsel_unsel.median(dim=1).values
                    else:
                        mean_d_unsel_to_own_unsel_excl_self = (
                            (d_unsel_unsel.sum(dim=1)) / (Kunsel - 1)
                        )
                else:
                    mean_d_unsel_to_own_unsel_excl_self = torch.zeros(Kunsel, device=device)

                # Multi-candidate anti-ms: top-N sel × top-N unsel simplified-delta search.
                _use_multicand = margin_swap_num_candidates > 1 and not margin_swap_vectorize_global_cdist
                if _use_multicand:
                    _mg_u_local = mean_d_unsel_to_other - mean_d_unsel_to_own_unsel_excl_self
                    _N_cand = min(margin_swap_num_candidates, Ksel, Kunsel)
                    _topk_s = (current_margin * float(objective_sign)).topk(_N_cand, largest=False).indices
                    _topk_u = (_mg_u_local * float(objective_sign)).topk(_N_cand, largest=True).indices
                    _best_mc_delta = 0.0
                    _best_mc_pair = None
                    for _si in _topk_s.tolist():
                        for _ui in _topk_u.tolist():
                            _d_eff = float((_mg_u_local[_ui] - current_margin[_si]) * float(objective_sign))
                            if _d_eff > _best_mc_delta + 1e-9:
                                _best_mc_delta = _d_eff
                                _best_mc_pair = (_si, _ui)
                    if _best_mc_pair is not None:
                        best = (_best_mc_delta * float(objective_sign), _best_mc_pair[0], _best_mc_pair[1])
                else:
                    if profile_anti_ms:
                        torch.cuda.synchronize(device); _t0 = _time.perf_counter()
                    for s_col in range(Ksel):
                        keep_mask = torch.ones(Ksel, dtype=torch.bool, device=device)
                        keep_mask[s_col] = False

                        add_to_kept = d_sel_sel[:, s_col:s_col + 1]
                        sub_from_kept = d_sel_unsel
                        new_mean_same_unsel_kept = (
                            Kunsel * mean_d_sel_to_own_unsel.unsqueeze(1)
                            + add_to_kept
                            - sub_from_kept
                        ) / float(Kunsel)
                        new_margin_kept = (
                            mean_d_sel_to_other.unsqueeze(1) - new_mean_same_unsel_kept
                        )
                        new_margin_kept = new_margin_kept.masked_fill(~keep_mask.unsqueeze(1), 0.0)
                        sum_kept_per_u = new_margin_kept.sum(dim=0)

                        if Kunsel >= 2:
                            new_mean_same_unsel_for_u = (
                                (Kunsel - 1) * mean_d_unsel_to_own_unsel_excl_self
                                + d_unsel_sel[:, s_col]
                            ) / float(Kunsel - 1)
                        else:
                            new_mean_same_unsel_for_u = torch.zeros(Kunsel, device=device)
                        new_margin_u = mean_d_unsel_to_other - new_mean_same_unsel_for_u

                        new_sum_per_u = sum_kept_per_u + new_margin_u
                        delta_per_u = new_sum_per_u - current_sum
                        if objective_sign > 0:
                            u_best = int(delta_per_u.argmax().item())
                            local_delta = float(delta_per_u[u_best].item())
                            if local_delta > 1e-9 and (best is None or local_delta > best[0]):
                                best = (local_delta, s_col, u_best)
                        else:
                            u_best = int(delta_per_u.argmin().item())
                            local_delta = float(delta_per_u[u_best].item())
                            if local_delta < -1e-9 and (best is None or local_delta < best[0]):
                                best = (local_delta, s_col, u_best)
                    if profile_anti_ms:
                        torch.cuda.synchronize(device)
                        _prof_totals["swap_search"] += _time.perf_counter() - _t0

                if best is not None:
                    _, s_col, u_col = best
                    old_global = int(member_idx[sel_local[s_col]].item())
                    new_global = int(unsel_global_for_swap[u_col].item())
                    selected_mask[old_global] = False
                    selected_mask[new_global] = True
                    any_improvement = True
                    if adaptive_n:
                        cluster_changed_this_pass[c] = True
                    if _pass1_cluster_deltas is not None and _pass == 0:
                        _pass1_cluster_deltas[c] = abs(best[0])
                    if margin_swap_vectorize_global_cdist:
                        if margin_swap_vectorize_mode == "B":
                            _needs_global_refresh = True
                        else:
                            g_dss, g_dns, g_sp, g_sc = _build_global_precompute()

                    # Chain-k=2 (2c): after applying first swap, search for a second swap
                    # on the updated cluster selection. Other-cluster selected is unchanged.
                    _chain_k_allowed = (
                        (_chain_eligible_set is not None and bool(_chain_eligible_set[c].item()))
                        if margin_swap_chain_k_top_pct > 0.0 else True
                    )
                    if _eff_chain_k >= 2 and not margin_swap_vectorize_global_cdist and _chain_k_allowed:
                        _cm = selected_mask[member_idx]
                        if _cm.any() and not _cm.all():
                            _c_sl = torch.nonzero(_cm, as_tuple=False).squeeze(-1)
                            _c_ul = torch.nonzero(~_cm, as_tuple=False).squeeze(-1)
                            _c_sf = member_feats.index_select(0, _c_sl)
                            _c_uf = member_feats.index_select(0, _c_ul)
                            _c_ug = member_idx.index_select(0, _c_ul)
                            _c_Ku = int(_c_ul.numel())
                            _c_Ks = int(_c_sl.numel())
                            if _c_Ku > 0:
                                # Reuse other-cluster distances (unchanged by within-cluster swap)
                                if _centroid_mode:
                                    _cd2s = torch.cdist(_c_sf, _chain_cent_k, p=2)
                                    _cd2u = torch.cdist(_c_uf, _chain_cent_k, p=2)
                                    _md_s2 = (_cd2s * _chain_w_k.unsqueeze(0)).sum(dim=1) / _chain_total_w
                                    _md_u2 = (_cd2u * _chain_w_k.unsqueeze(0)).sum(dim=1) / _chain_total_w
                                else:
                                    _osf2 = other_sel_feats
                                    _md_s2 = _agg(torch.cdist(_c_sf, _osf2, p=2))
                                    _md_u2 = _agg(torch.cdist(_c_uf, _osf2, p=2))
                                _ss2 = torch.cdist(_c_sf, _c_sf, p=2)
                                _su2 = torch.cdist(_c_sf, _c_uf, p=2)
                                _uu2 = torch.cdist(_c_uf, _c_uf, p=2)
                                _mso2 = _su2.mean(dim=1)
                                _cm2 = _md_s2 - _mso2
                                _cs2 = float(_cm2.sum().item())
                                _mu2 = (_uu2.sum(dim=1) / (_c_Ku - 1)) if _c_Ku >= 2 else torch.zeros(_c_Ku, device=device)
                                _best2 = None
                                for _s2 in range(_c_Ks):
                                    _km2 = torch.ones(_c_Ks, dtype=torch.bool, device=device)
                                    _km2[_s2] = False
                                    _nm2 = (_c_Ku * _mso2.unsqueeze(1) + _ss2[:, _s2:_s2+1] - _su2) / float(_c_Ku)
                                    _nmg2 = (_md_s2.unsqueeze(1) - _nm2).masked_fill(~_km2.unsqueeze(1), 0.0)
                                    _sk2 = _nmg2.sum(dim=0)
                                    _nmu2 = ((_c_Ku-1)*_mu2 + _su2.t()[:, _s2]) / float(_c_Ku-1) if _c_Ku >= 2 else torch.zeros(_c_Ku, device=device)
                                    _d2 = _sk2 + (_md_u2 - _nmu2) - _cs2
                                    if objective_sign < 0:
                                        _ub2 = int(_d2.argmin().item())
                                        _ld2 = float(_d2[_ub2].item())
                                        if _ld2 < -1e-9 and (_best2 is None or _ld2 < _best2[0]):
                                            _best2 = (_ld2, _s2, _ub2)
                                    else:
                                        _ub2 = int(_d2.argmax().item())
                                        _ld2 = float(_d2[_ub2].item())
                                        if _ld2 > 1e-9 and (_best2 is None or _ld2 > _best2[0]):
                                            _best2 = (_ld2, _s2, _ub2)
                                if _best2 is not None:
                                    _, _s2c, _u2c = _best2
                                    selected_mask[int(member_idx[_c_sl[_s2c]].item())] = False
                                    selected_mask[int(_c_ug[_u2c].item())] = True

            if not any_improvement:
                break
            # After pass 0, build chain-k eligible set from top-pct deltas.
            if _pass1_cluster_deltas is not None and _pass == 0:
                _n_elig = max(1, int(math.ceil(float(num_clusters) * margin_swap_chain_k_top_pct)))
                _top_vals, _ = _pass1_cluster_deltas.topk(_n_elig, largest=True, sorted=False)
                _thresh = float(_top_vals.min().item())
                _chain_eligible_set = _pass1_cluster_deltas >= _thresh
            if adaptive_n:
                cluster_active = cluster_changed_this_pass
            # Strategy B: refresh global matrices once per pass boundary.
            if margin_swap_vectorize_global_cdist and margin_swap_vectorize_mode == "B" and _needs_global_refresh:
                g_dss, g_dns, g_sp, g_sc = _build_global_precompute()

    # Post-antims max-spread passes (2d): after anti-ms converges, run additional
    # per-cluster greedy swaps maximizing mean distance to current selected pool.
    if post_antims_max_spread_k > 0:
        for _sp in range(post_antims_max_spread_k):
            for c in range(num_clusters):
                _mm_s = labels_flat == c
                _mi_s = torch.nonzero(_mm_s, as_tuple=False).squeeze(-1)
                if _mi_s.numel() < margin_swap_min_cluster_size:
                    continue
                _csm_s = selected_mask[_mi_s]
                if not _csm_s.any() or _csm_s.all():
                    continue
                _sl_s = torch.nonzero(_csm_s, as_tuple=False).squeeze(-1)
                _ul_s = torch.nonzero(~_csm_s, as_tuple=False).squeeze(-1)
                _sf_s = samples.index_select(0, _mi_s.index_select(0, _sl_s))
                _uf_s = samples.index_select(0, _mi_s.index_select(0, _ul_s))
                _all_sel_s = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
                _asf_s = samples.index_select(0, _all_sel_s)
                _S = int(_all_sel_s.numel())
                # Mean dist from sel patches to all selected (including self → small bias OK)
                _d_s2a = torch.cdist(_sf_s, _asf_s, p=2)
                _mean_s = _d_s2a.sum(dim=1) / max(_S - 1, 1)
                # Mean dist from unsel patches to all selected
                _d_u2a = torch.cdist(_uf_s, _asf_s, p=2)
                _mean_u = _d_u2a.sum(dim=1) / max(_S, 1)
                _best_sel = int(_mean_s.argmin().item())
                _best_unsel = int(_mean_u.argmax().item())
                if float(_mean_u[_best_unsel].item()) > float(_mean_s[_best_sel].item()) + 1e-9:
                    _og_s = int(_mi_s[_sl_s[_best_sel]].item())
                    _ng_s = int(_mi_s[_ul_s[_best_unsel]].item())
                    selected_mask[_og_s] = False
                    selected_mask[_ng_s] = True

    # Profile: print per-op totals across all passes+clusters for this slide.
    if profile_anti_ms:
        torch.cuda.synchronize(device)
        for _pk, _pv in _prof_totals.items():
            print(f"[ANTI_MS_PROFILE] {_pk} {_pv * 1000.0:.2f}", flush=True)

    return torch.nonzero(selected_mask, as_tuple=False).squeeze(-1).to(dtype=torch.int64)


# =============================================================================
# Global k-NN bounded swap refinement.
# Complements per-cluster anti-ms by catching cross-cluster swaps: for each
# selected patch s, tests swapping s with each of its k_nn nearest globally-
# unselected neighbors under a mean-dist-to-other-selected objective.
# Single-pass greedy: finds the globally best improving swap and applies it.
# Preserves |S| exactly.
# =============================================================================


@torch.no_grad()
def knn_swap_refine(
    samples: torch.Tensor,
    labels: torch.Tensor,  # interface parity with swap_refine_margin_per_cluster
    selected_indices: torch.Tensor,
    k_nn: int = 50,
    objective_sign: int = -1,
) -> torch.Tensor:
    """Single-pass global k-NN swap refinement.

    For each selected patch s_i, finds its k_nn nearest unselected neighbors.
    Evaluates swapping s_i with each candidate under a mean-dist-to-other-
    selected objective (same spirit as anti-ms but globally bounded).
    Applies the single best improving swap; preserves |S| exactly.

    objective_sign=-1 (anti-ms analog): minimize mean dist to other selected
    (make selected patches cluster together).
    objective_sign=+1: maximize mean dist (diversity; closed on all tested cells).
    """
    if k_nn <= 0 or selected_indices.numel() < 2:
        return selected_indices

    device = samples.device
    N = samples.size(0)

    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    selected_mask[selected_indices] = True

    sel_idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
    unsel_idx = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
    if unsel_idx.numel() == 0 or sel_idx.numel() < 2:
        return selected_indices

    sel_feats = samples.index_select(0, sel_idx)     # (S, D)
    unsel_feats = samples.index_select(0, unsel_idx)  # (U, D)
    S = int(sel_feats.size(0))

    # Pairwise distances among selected: (S, S)  diagonal = 0
    d_sel_sel = torch.cdist(sel_feats, sel_feats, p=2)
    # Distances from selected to unselected: (S, U)
    d_sel_unsel = torch.cdist(sel_feats, unsel_feats, p=2)

    # sum_d_to_others[i] = sum_j d[i,j] (diagonal=0, so equals sum_{j≠i} d[i,j])
    sum_d_sel_to_others = d_sel_sel.sum(dim=1)  # (S,)
    mean_d_sel_to_others = sum_d_sel_to_others / float(S - 1)  # (S,)

    # For each selected, find k nearest unselected neighbors
    k_eff = min(k_nn, int(unsel_idx.numel()))
    _, knn_local = d_sel_unsel.topk(k=k_eff, largest=False, dim=1)  # (S, k)

    # Find globally best improving swap.
    # Exact delta formula (derived from mean-dist-to-others total):
    # delta = 2 × (mean_d_cand_to_others − mean_d_sel_to_others[s_col])
    # where mean_d_cand_to_others = (sum dist from cand to sel[j≠s_col]) / (S-1)
    best_delta = 0.0
    best_s_col = -1
    best_u_local = -1

    for s_col in range(S):
        cand_local_idx = knn_local[s_col]  # (k,) into unsel_idx
        cand_feats = unsel_feats.index_select(0, cand_local_idx)  # (k, D)

        # dist from each candidate to all selected: (k, S)
        d_cand_sel = torch.cdist(cand_feats, sel_feats, p=2)
        # dist from each candidate to (sel − {s_col}): exclude col s_col
        sum_d_cand_to_others = d_cand_sel.sum(dim=1) - d_cand_sel[:, s_col]  # (k,)
        mean_d_cand_to_others = sum_d_cand_to_others / float(S - 1)  # (k,)

        cur_mean = float(mean_d_sel_to_others[s_col].item())
        # delta in total mean_d_to_others when swapping s_col ↔ cand[j]:
        # = 2 × (mean_d_cand_to_others[j] − cur_mean)
        delta_per_cand = 2.0 * (mean_d_cand_to_others - cur_mean)  # (k,)

        if objective_sign > 0:
            k_best = int(delta_per_cand.argmax().item())
            delta = float(delta_per_cand[k_best].item())
            if delta > 1e-9 and delta > best_delta:
                best_delta = delta
                best_s_col = s_col
                best_u_local = int(cand_local_idx[k_best].item())
        else:  # anti-ms: minimize
            k_best = int(delta_per_cand.argmin().item())
            delta = float(delta_per_cand[k_best].item())
            if delta < -1e-9 and delta < best_delta:
                best_delta = delta
                best_s_col = s_col
                best_u_local = int(cand_local_idx[k_best].item())

    if best_s_col >= 0:
        old_global = int(sel_idx[best_s_col].item())
        new_global = int(unsel_idx[best_u_local].item())
        selected_mask[old_global] = False
        selected_mask[new_global] = True

    return torch.nonzero(selected_mask, as_tuple=False).squeeze(-1).to(dtype=torch.int64)
