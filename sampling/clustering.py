from __future__ import annotations

import math
import torch

from .rapids import build_kmeans, cupy_to_torch, ensure_cupy_array, torch_to_cupy

# Module-level compiled cdist helper (CDIST_COMPILE=1 gate). Lazily initialized on first use.
_COMPILED_CDIST: object | None = None


def _cdist_precompute(
    samples: torch.Tensor,
    centers: torch.Tensor,
    means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor cdist precompute body; target for torch.compile."""
    return (
        torch.cdist(samples, centers, p=2).square_(),
        torch.cdist(samples, means, p=2).square_(),
    )


def resolve_num_clusters(
    num_samples: int,
    ratio: float,
    min_clusters: int,
    target_selected: int | None,
) -> int:
    num_clusters = max(int(ratio * num_samples), min_clusters)
    num_clusters = min(num_clusters, num_samples)
    if target_selected is not None:
        num_clusters = min(num_clusters, target_selected)
    if not 1 <= num_clusters <= num_samples:
        raise ValueError(
            f"Invalid num_clusters=int({ratio} * {num_samples})={num_clusters}. "
            f"Expected 1 <= K <= N."
        )
    return num_clusters


def resolve_target_selected(num_samples: int, budget_ratio: float | None) -> int | None:
    if budget_ratio is None:
        return None
    target_selected = int(budget_ratio * num_samples)
    if not 1 <= target_selected <= num_samples:
        raise ValueError(
            f"Invalid target_selected=int({budget_ratio} * {num_samples})={target_selected}. "
            f"Expected 1 <= target_selected <= N."
        )
    return target_selected


def resolve_effective_ratio(
    num_samples: int,
    base_ratio: float,
    adaptive_ratio_schedule: list[tuple[int | None, float]],
) -> float:
    for max_samples, ratio in adaptive_ratio_schedule:
        if max_samples is None or num_samples <= max_samples:
            return ratio
    return base_ratio


@torch.no_grad()
def find_global_nearest_indices(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    sample_block_size: int,
) -> torch.Tensor:
    num_samples = samples.size(0)
    num_centroids = centroids.size(0)
    device = samples.device

    best_dist = torch.full((num_centroids,), float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.full((num_centroids,), num_samples, device=device, dtype=torch.int64)
    centroid_norm = centroids.square().sum(dim=1).view(1, -1)

    for start in range(0, num_samples, sample_block_size):
        stop = min(start + sample_block_size, num_samples)
        block = samples[start:stop]
        block_norm = block.square().sum(dim=1, keepdim=True)
        dist = block_norm - 2.0 * (block @ centroids.T) + centroid_norm
        dist = dist.clamp_min_(0.0)

        block_best_dist, block_best_local_idx = dist.min(dim=0)
        block_best_idx = block_best_local_idx.to(torch.int64) + start

        update = (block_best_dist < best_dist) | (
            (block_best_dist == best_dist) & (block_best_idx < best_idx)
        )
        best_dist = torch.where(update, block_best_dist, best_dist)
        best_idx = torch.where(update, block_best_idx, best_idx)

    return best_idx


@torch.no_grad()
def select_representative_indices(
    samples: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    sample_block_size: int,
) -> tuple[torch.Tensor, int]:
    num_samples = samples.size(0)
    num_clusters = centroids.size(0)
    device = samples.device

    labels = labels.reshape(-1).to(dtype=torch.int64, device=device)
    if labels.numel() != num_samples:
        raise ValueError(
            f"Label count mismatch: got {labels.numel()} labels for {num_samples} samples."
        )

    best_dist = torch.full((num_clusters,), float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.full((num_clusters,), num_samples, device=device, dtype=torch.int64)

    for start in range(0, num_samples, sample_block_size):
        stop = min(start + sample_block_size, num_samples)
        block = samples[start:stop]
        block_labels = labels[start:stop]
        block_centroids = centroids.index_select(0, block_labels)

        block_dist = (
            block.square().sum(dim=1)
            + block_centroids.square().sum(dim=1)
            - 2.0 * (block * block_centroids).sum(dim=1)
        ).clamp_min_(0.0)

        local_best_dist = torch.full(
            (num_clusters,), float("inf"), device=device, dtype=torch.float32
        )
        local_best_dist.scatter_reduce_(
            0, block_labels, block_dist, reduce="amin", include_self=True
        )

        block_indices = torch.arange(start, stop, device=device, dtype=torch.int64)
        sentinel = torch.full((stop - start,), num_samples, device=device, dtype=torch.int64)
        is_local_best = block_dist <= local_best_dist.index_select(0, block_labels)
        local_candidates = torch.where(is_local_best, block_indices, sentinel)

        local_best_idx = torch.full(
            (num_clusters,), num_samples, device=device, dtype=torch.int64
        )
        local_best_idx.scatter_reduce_(
            0, block_labels, local_candidates, reduce="amin", include_self=True
        )

        update = (local_best_dist < best_dist) | (
            (local_best_dist == best_dist) & (local_best_idx < best_idx)
        )
        best_dist = torch.where(update, local_best_dist, best_dist)
        best_idx = torch.where(update, local_best_idx, best_idx)

    counts = torch.bincount(labels, minlength=num_clusters)
    used_clusters = int((counts > 0).sum().item())
    empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten()
    if empty_clusters.numel() > 0:
        fallback_idx = find_global_nearest_indices(
            samples=samples,
            centroids=centroids.index_select(0, empty_clusters),
            sample_block_size=sample_block_size,
        )
        best_idx.index_copy_(0, empty_clusters, fallback_idx)

    return best_idx, used_clusters


@torch.no_grad()
def assign_to_centers(
    samples: torch.Tensor,
    centers: torch.Tensor,
    sample_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = samples.size(0)
    center_norm = centers.square().sum(dim=1).view(1, -1)

    labels_chunks = []
    min_dist_chunks = []
    for start in range(0, num_samples, sample_block_size):
        stop = min(start + sample_block_size, num_samples)
        block = samples[start:stop]
        block_norm = block.square().sum(dim=1, keepdim=True)
        dist = block_norm - 2.0 * (block @ centers.T) + center_norm
        dist = dist.clamp_min_(0.0)
        block_min_dist, block_labels = dist.min(dim=1)
        labels_chunks.append(block_labels.to(dtype=torch.int64))
        min_dist_chunks.append(block_min_dist)

    return torch.cat(labels_chunks, dim=0), torch.cat(min_dist_chunks, dim=0)


@torch.no_grad()
def select_bounded_l2_medoid_index(
    samples: torch.Tensor,
    member_indices: torch.Tensor,
    mean: torch.Tensor,
    candidate_size: int,
    candidate_strategy: str,
    current_centers: torch.Tensor,
    cluster_id: int,
    sample_block_size: int,
    centroid_reg: float,
    separation_reg: float,
    knn_mean_margin_pre_k_mult: float = 4.0,
    knn_mean_margin_score_mode: str = "ratio",
    knn_mean_margin_subtractive_lambda: float = 1.0,
    refine_distance_metric: str = "l2",
    margin_dir_1d: bool = False,
) -> torch.Tensor:
    """Bounded within-cluster L2 medoid with optional margin shortlist.

    candidate_strategy ∈ {"mean", "mean_margin"}. The "fps", "mean_fps",
    "mean_margin_ratio", and "max_coverage" variants were closed cycles 3-28.
    """
    member_feats = samples.index_select(0, member_indices)
    count = int(member_feats.size(0))
    if count <= 1:
        return member_indices[0]

    if refine_distance_metric == "cosine":
        import torch.nn.functional as _F
        _eps = 1e-8
        member_feats = _F.normalize(member_feats, p=2, dim=-1, eps=_eps)
        mean = _F.normalize(mean.view(1, -1), p=2, dim=-1, eps=_eps).squeeze(0)
        if current_centers.size(0) > 0:
            current_centers = _F.normalize(current_centers, p=2, dim=-1, eps=_eps)

    mean_sq_dist = (member_feats - mean.view(1, -1)).square().sum(dim=1)

    if candidate_size > 0 and candidate_size < count:
        if candidate_strategy == "mean":
            candidate_local = torch.topk(
                mean_sq_dist, k=candidate_size, largest=False, sorted=False,
            ).indices
        elif candidate_strategy == "mean_margin":
            if current_centers.size(0) <= 1:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                other_mask = torch.ones(
                    current_centers.size(0), device=samples.device, dtype=torch.bool
                )
                other_mask[cluster_id] = False
                competing_centers = current_centers[other_mask]
                competing_sq_dist = torch.cdist(member_feats, competing_centers, p=2).square()
                nearest_competing_sq_dist = competing_sq_dist.min(dim=1).values.clamp_min_(1e-12)
                margin_score = mean_sq_dist / nearest_competing_sq_dist
                candidate_local = torch.topk(
                    margin_score, k=candidate_size, largest=False, sorted=False,
                ).indices
        elif candidate_strategy == "margin_max":
            # margin_max: pick candidates maximizing (competing_dist - own_dist).
            # Selects the most discriminative boundary patches within the cluster.
            if current_centers.size(0) <= 1:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                other_mask = torch.ones(
                    current_centers.size(0), device=samples.device, dtype=torch.bool
                )
                other_mask[cluster_id] = False
                competing_centers = current_centers[other_mask]
                if margin_dir_1d:
                    # 1D margin-direction projection.
                    own_ctr = current_centers[cluster_id].view(1, -1)
                    d_oc = torch.cdist(own_ctr, competing_centers, p=2).squeeze(0)
                    nearest_idx = int(d_oc.argmin().item())
                    nearest_ctr = competing_centers[nearest_idx].view(1, -1)
                    direction = (own_ctr - nearest_ctr).squeeze(0)
                    direction = direction / direction.norm().clamp_min_(1e-8)
                    proj_m = member_feats @ direction
                    proj_own = float((own_ctr @ direction).squeeze())
                    proj_near = float((nearest_ctr @ direction).squeeze())
                    own_sq_1d = (proj_m - proj_own).square()
                    near_sq_1d = (proj_m - proj_near).square().clamp_min_(1e-12)
                    margin_score = near_sq_1d - own_sq_1d
                else:
                    competing_sq_dist = torch.cdist(member_feats, competing_centers, p=2).square()
                    nearest_competing_sq_dist = competing_sq_dist.min(dim=1).values
                    margin_score = nearest_competing_sq_dist - mean_sq_dist
                candidate_local = torch.topk(
                    margin_score, k=candidate_size, largest=True, sorted=False,
                ).indices
        elif candidate_strategy == "knn_mean_margin":
            # Pre-shortlist by mean_sq_dist using configurable mult, then margin-score.
            pre_k = min(max(int(round(candidate_size * knn_mean_margin_pre_k_mult)), candidate_size), count)
            pre_shortlist = torch.topk(
                mean_sq_dist, k=pre_k, largest=False, sorted=False,
            ).indices
            pre_feats = member_feats.index_select(0, pre_shortlist)
            pre_mean_sq = mean_sq_dist.index_select(0, pre_shortlist)
            if current_centers.size(0) <= 1:
                inner = torch.topk(
                    pre_mean_sq, k=candidate_size, largest=False, sorted=False,
                ).indices
                candidate_local = pre_shortlist.index_select(0, inner)
            else:
                other_mask = torch.ones(
                    current_centers.size(0), device=samples.device, dtype=torch.bool
                )
                other_mask[cluster_id] = False
                competing_centers = current_centers[other_mask]
                if margin_dir_1d:
                    # 1D margin-direction projection for knn_mean_margin.
                    own_ctr = current_centers[cluster_id].view(1, -1)
                    d_oc = torch.cdist(own_ctr, competing_centers, p=2).squeeze(0)
                    nearest_idx = int(d_oc.argmin().item())
                    nearest_ctr = competing_centers[nearest_idx].view(1, -1)
                    direction = (own_ctr - nearest_ctr).squeeze(0)
                    direction = direction / direction.norm().clamp_min_(1e-8)
                    proj_pre = pre_feats @ direction
                    proj_own = float((own_ctr @ direction).squeeze())
                    proj_near = float((nearest_ctr @ direction).squeeze())
                    own_sq_1d = (proj_pre - proj_own).square()
                    near_sq_1d = (proj_pre - proj_near).square().clamp_min_(1e-12)
                    margin_score = own_sq_1d / near_sq_1d
                else:
                    competing_sq_dist = torch.cdist(pre_feats, competing_centers, p=2).square()
                    nearest_competing_sq_dist = competing_sq_dist.min(dim=1).values.clamp_min_(1e-12)
                    if knn_mean_margin_score_mode == "subtractive":
                        margin_score = pre_mean_sq - knn_mean_margin_subtractive_lambda * nearest_competing_sq_dist
                    else:
                        margin_score = pre_mean_sq / nearest_competing_sq_dist
                inner = torch.topk(
                    margin_score, k=candidate_size, largest=False, sorted=False,
                ).indices
                candidate_local = pre_shortlist.index_select(0, inner)
        elif candidate_strategy == "local_density":
            # Pick candidates at intra-cluster density peaks: k=5 NN mean-distance minimizers.
            k_density = min(5, count - 1)
            if k_density <= 0:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                intra_dist = torch.cdist(member_feats, member_feats, p=2)
                intra_dist.fill_diagonal_(float("inf"))
                knn_mean = intra_dist.topk(k=k_density, largest=False).values.mean(dim=1)
                candidate_local = torch.topk(
                    knn_mean, k=candidate_size, largest=False, sorted=False,
                ).indices
        else:
            raise ValueError(
                f"Unknown kmedoid_refine_candidate_strategy: {candidate_strategy!r}"
            )

        candidate_indices = member_indices.index_select(0, candidate_local)
        candidate_feats = member_feats.index_select(0, candidate_local)
        candidate_mean_sq_dist = mean_sq_dist.index_select(0, candidate_local)
    else:
        candidate_indices = member_indices
        candidate_feats = member_feats
        candidate_mean_sq_dist = mean_sq_dist

    candidate_sq_norm = candidate_feats.square().sum(dim=1, keepdim=True)
    objective = torch.zeros(candidate_feats.size(0), device=samples.device, dtype=torch.float32)
    for start in range(0, count, sample_block_size):
        stop = min(start + sample_block_size, count)
        block_feats = member_feats[start:stop]
        block_sq_norm = block_feats.square().sum(dim=1).view(1, -1)
        sq_dist = candidate_sq_norm + block_sq_norm - 2.0 * (candidate_feats @ block_feats.T)
        objective += sq_dist.clamp_min_(0.0).sum(dim=1)

    if centroid_reg > 0:
        objective += float(centroid_reg) * float(count) * candidate_mean_sq_dist

    if separation_reg > 0 and current_centers.size(0) > 1:
        other_mask = torch.ones(
            current_centers.size(0), device=samples.device, dtype=torch.bool
        )
        other_mask[cluster_id] = False
        competing_centers = current_centers[other_mask]
        competing_sq_dist = torch.cdist(candidate_feats, competing_centers, p=2).square()
        nearest_competing_sq_dist = competing_sq_dist.min(dim=1).values
        objective -= float(separation_reg) * float(count) * nearest_competing_sq_dist

    return candidate_indices[objective.argmin()]


@torch.no_grad()
def _select_bounded_l2_medoid_index_from_precomputed(
    samples: torch.Tensor,
    member_indices: torch.Tensor,
    count: int,
    cluster_id: int,
    samples_to_means_sq: torch.Tensor,
    samples_to_centers_sq: torch.Tensor,
    candidate_size: int,
    candidate_strategy: str,
    sample_block_size: int,
    centroid_reg: float,
    separation_reg: float,
    knn_mean_margin_pre_k_mult: float = 4.0,
    knn_mean_margin_score_mode: str = "ratio",
    knn_mean_margin_subtractive_lambda: float = 1.0,
    margin_dir_1d: bool = False,
    current_centers: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Vectorized variant of select_bounded_l2_medoid_index using precomputed distance matrices.

    Uses samples_to_means_sq[N,K] and samples_to_centers_sq[N,K] precomputed once per refine
    iteration in update_medoid_indices_from_assignments, eliminating per-cluster cdist launches.
    """
    if count <= 1:
        return member_indices[0]

    member_feats = samples.index_select(0, member_indices)
    mean_sq_dist = samples_to_means_sq.index_select(0, member_indices)[:, cluster_id]

    num_centers = samples_to_centers_sq.size(1)

    if candidate_size > 0 and candidate_size < count:
        if candidate_strategy == "mean":
            candidate_local = torch.topk(
                mean_sq_dist, k=candidate_size, largest=False, sorted=False,
            ).indices
        elif candidate_strategy == "mean_margin":
            if num_centers <= 1:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                other_mask = torch.ones(num_centers, device=samples.device, dtype=torch.bool)
                other_mask[cluster_id] = False
                member_to_others_sq = samples_to_centers_sq.index_select(0, member_indices)[:, other_mask]
                nearest_competing_sq_dist = member_to_others_sq.min(dim=1).values.clamp_min_(1e-12)
                margin_score = mean_sq_dist / nearest_competing_sq_dist
                candidate_local = torch.topk(
                    margin_score, k=candidate_size, largest=False, sorted=False,
                ).indices
        elif candidate_strategy == "margin_max":
            if num_centers <= 1:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                other_mask = torch.ones(num_centers, device=samples.device, dtype=torch.bool)
                other_mask[cluster_id] = False
                member_to_others_sq = samples_to_centers_sq.index_select(0, member_indices)[:, other_mask]
                if margin_dir_1d and current_centers is not None:
                    # 1D margin-direction projection (precomputed path).
                    own_ctr = current_centers[cluster_id].view(1, -1)
                    other_ctrs = current_centers[other_mask]
                    d_oc = torch.cdist(own_ctr, other_ctrs, p=2).squeeze(0)
                    nearest_idx = int(d_oc.argmin().item())
                    nearest_ctr = other_ctrs[nearest_idx].view(1, -1)
                    direction = (own_ctr - nearest_ctr).squeeze(0)
                    direction = direction / direction.norm().clamp_min_(1e-8)
                    proj_m = member_feats @ direction
                    proj_own = float((own_ctr @ direction).squeeze())
                    proj_near = float((nearest_ctr @ direction).squeeze())
                    own_sq_1d = (proj_m - proj_own).square()
                    near_sq_1d = (proj_m - proj_near).square().clamp_min_(1e-12)
                    margin_score = near_sq_1d - own_sq_1d
                else:
                    nearest_competing_sq_dist = member_to_others_sq.min(dim=1).values
                    margin_score = nearest_competing_sq_dist - mean_sq_dist
                candidate_local = torch.topk(
                    margin_score, k=candidate_size, largest=True, sorted=False,
                ).indices
        elif candidate_strategy == "knn_mean_margin":
            pre_k = min(max(int(round(candidate_size * knn_mean_margin_pre_k_mult)), candidate_size), count)
            pre_shortlist = torch.topk(
                mean_sq_dist, k=pre_k, largest=False, sorted=False,
            ).indices
            pre_mean_sq = mean_sq_dist.index_select(0, pre_shortlist)
            if num_centers <= 1:
                inner = torch.topk(
                    pre_mean_sq, k=candidate_size, largest=False, sorted=False,
                ).indices
                candidate_local = pre_shortlist.index_select(0, inner)
            else:
                other_mask = torch.ones(num_centers, device=samples.device, dtype=torch.bool)
                other_mask[cluster_id] = False
                pre_member_indices = member_indices.index_select(0, pre_shortlist)
                pre_to_others_sq = samples_to_centers_sq.index_select(0, pre_member_indices)[:, other_mask]
                if margin_dir_1d and current_centers is not None:
                    # 1D margin-direction projection for knn_mean_margin (precomputed path).
                    own_ctr = current_centers[cluster_id].view(1, -1)
                    other_ctrs = current_centers[other_mask]
                    d_oc = torch.cdist(own_ctr, other_ctrs, p=2).squeeze(0)
                    nearest_idx = int(d_oc.argmin().item())
                    nearest_ctr = other_ctrs[nearest_idx].view(1, -1)
                    direction = (own_ctr - nearest_ctr).squeeze(0)
                    direction = direction / direction.norm().clamp_min_(1e-8)
                    pre_feats_1d = samples.index_select(0, pre_member_indices)
                    proj_pre = pre_feats_1d @ direction
                    proj_own = float((own_ctr @ direction).squeeze())
                    proj_near = float((nearest_ctr @ direction).squeeze())
                    own_sq_1d = (proj_pre - proj_own).square()
                    near_sq_1d = (proj_pre - proj_near).square().clamp_min_(1e-12)
                    margin_score = own_sq_1d / near_sq_1d
                else:
                    nearest_competing_sq_dist = pre_to_others_sq.min(dim=1).values.clamp_min_(1e-12)
                    if knn_mean_margin_score_mode == "subtractive":
                        margin_score = pre_mean_sq - knn_mean_margin_subtractive_lambda * nearest_competing_sq_dist
                    else:
                        margin_score = pre_mean_sq / nearest_competing_sq_dist
                inner = torch.topk(
                    margin_score, k=candidate_size, largest=False, sorted=False,
                ).indices
                candidate_local = pre_shortlist.index_select(0, inner)
        elif candidate_strategy == "local_density":
            # Pick candidates at intra-cluster density peaks: k=5 NN mean-distance minimizers.
            k_density = min(5, count - 1)
            if k_density <= 0:
                candidate_local = torch.topk(
                    mean_sq_dist, k=candidate_size, largest=False, sorted=False,
                ).indices
            else:
                intra_dist = torch.cdist(member_feats, member_feats, p=2)
                intra_dist.fill_diagonal_(float("inf"))
                knn_mean = intra_dist.topk(k=k_density, largest=False).values.mean(dim=1)
                candidate_local = torch.topk(
                    knn_mean, k=candidate_size, largest=False, sorted=False,
                ).indices
        else:
            raise ValueError(
                f"Unknown kmedoid_refine_candidate_strategy: {candidate_strategy!r}"
            )

        candidate_indices = member_indices.index_select(0, candidate_local)
        candidate_feats = member_feats.index_select(0, candidate_local)
        candidate_mean_sq_dist = mean_sq_dist.index_select(0, candidate_local)
    else:
        candidate_indices = member_indices
        candidate_feats = member_feats
        candidate_mean_sq_dist = mean_sq_dist

    candidate_sq_norm = candidate_feats.square().sum(dim=1, keepdim=True)
    objective = torch.zeros(candidate_feats.size(0), device=samples.device, dtype=torch.float32)
    for start in range(0, count, sample_block_size):
        stop = min(start + sample_block_size, count)
        block_feats = member_feats[start:stop]
        block_sq_norm = block_feats.square().sum(dim=1).view(1, -1)
        sq_dist = candidate_sq_norm + block_sq_norm - 2.0 * (candidate_feats @ block_feats.T)
        objective += sq_dist.clamp_min_(0.0).sum(dim=1)

    if centroid_reg > 0:
        objective += float(centroid_reg) * float(count) * candidate_mean_sq_dist

    if separation_reg > 0 and num_centers > 1:
        other_mask = torch.ones(num_centers, device=samples.device, dtype=torch.bool)
        other_mask[cluster_id] = False
        candidate_to_others_sq = samples_to_centers_sq.index_select(0, candidate_indices)[:, other_mask]
        nearest_competing_sq_dist = candidate_to_others_sq.min(dim=1).values
        objective -= float(separation_reg) * float(count) * nearest_competing_sq_dist

    return candidate_indices[objective.argmin()]


def _squared_cdist_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise L2 distances via matmul identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T."""
    a_sq = a.square().sum(dim=1, keepdim=True)
    b_sq = b.square().sum(dim=1).unsqueeze(0)
    return (a_sq + b_sq - 2.0 * (a @ b.T)).clamp_min_(0.0)


@torch.no_grad()
def update_medoid_indices_from_assignments(
    samples: torch.Tensor,
    labels: torch.Tensor,
    current_centers: torch.Tensor,
    num_clusters: int,
    sample_block_size: int,
    refine_candidate_size: int,
    refine_candidate_strategy: str,
    refine_min_cluster_size: int,
    refine_centroid_reg: float,
    refine_separation_reg: float,
    cdist_impl: str = "torch",
    cdist_fp16: bool = False,
    cdist_compile: bool = False,
    refine_batched: bool = False,
    refine_candidate_size_mode: str = "fixed",
    refine_candidate_size_floor: int = 4,
    refine_candidate_size_cap: int = 16,
    knn_mean_margin_pre_k_mult: float = 4.0,
    knn_mean_margin_score_mode: str = "ratio",
    knn_mean_margin_subtractive_lambda: float = 1.0,
    refine_separation_reg_mode: str = "fixed",
    refine_separation_reg_scale_min: float = 0.5,
    refine_separation_reg_scale_max: float = 2.0,
    refine_strategy_large_cluster: str = "",
    refine_strategy_size_adaptive_threshold: int = 0,
    margin_dir_1d: bool = False,
    sep_reg_density_per_k: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = samples.size(0)
    device = samples.device

    sums = samples.new_zeros((num_clusters, samples.size(1)))
    sums.index_add_(0, labels, samples)

    counts = torch.bincount(labels, minlength=num_clusters)
    nonempty = counts > 0

    means = sums.clone()
    if nonempty.any():
        means[nonempty] /= counts[nonempty].to(samples.dtype).unsqueeze(1)

    order = torch.argsort(labels)
    best_idx = torch.full((num_clusters,), num_samples, device=device, dtype=torch.int64)
    assigned_means = means.index_select(0, labels)
    sample_dist = (samples - assigned_means).square().sum(dim=1)

    best_dist = torch.full((num_clusters,), float("inf"), device=device, dtype=torch.float32)
    best_dist.scatter_reduce_(0, labels, sample_dist, reduce="amin", include_self=True)

    sample_indices = torch.arange(num_samples, device=device, dtype=torch.int64)
    sentinel = torch.full((num_samples,), num_samples, device=device, dtype=torch.int64)
    is_best = sample_dist <= best_dist.index_select(0, labels)
    candidates = torch.where(is_best, sample_indices, sentinel)
    best_idx.scatter_reduce_(0, labels, candidates, reduce="amin", include_self=True)

    if refine_candidate_size <= 0:
        return best_idx, means, counts

    # Precompute sample-to-center and sample-to-mean squared distances once per call.
    # Eliminates K per-cluster cdist kernel launches in the loop, replacing them with
    # 2 fused (N,K) matmuls. Shape (N,K); K-proportional Python-loop overhead removed.
    if cdist_fp16:
        samples_h = samples.to(dtype=torch.float16)
        centers_h = current_centers.to(dtype=torch.float16)
        means_h = means.to(dtype=torch.float16)
        samples_to_centers_sq = torch.cdist(samples_h, centers_h, p=2).square_().float().clamp_min_(0.0)
        samples_to_means_sq = torch.cdist(samples_h, means_h, p=2).square_().float().clamp_min_(0.0)
    elif cdist_impl == "matmul":
        samples_to_centers_sq = _squared_cdist_matmul(samples, current_centers)
        samples_to_means_sq = _squared_cdist_matmul(samples, means)
    elif cdist_compile:
        global _COMPILED_CDIST
        if _COMPILED_CDIST is None:
            _COMPILED_CDIST = torch.compile(_cdist_precompute, mode="reduce-overhead", dynamic=True)
        try:
            samples_to_centers_sq, samples_to_means_sq = _COMPILED_CDIST(samples, current_centers, means)
        except Exception:
            samples_to_centers_sq = torch.cdist(samples, current_centers, p=2).square_()
            samples_to_means_sq = torch.cdist(samples, means, p=2).square_()
    else:
        samples_to_centers_sq = torch.cdist(samples, current_centers, p=2).square_()
        samples_to_means_sq = torch.cdist(samples, means, p=2).square_()

    # Pull counts to CPU once to avoid K per-cluster .item() syncs in the loop.
    counts_list: list[int] = counts.tolist()
    num_centers = samples_to_centers_sq.size(1)

    if refine_batched:
        # Algebraic-identity batched path: eliminates per-cluster within-cluster cdist launches.
        # Identity: sum_i ||c_j - m_i||^2 = M * ||c_j - mean||^2 + sum_i ||m_i - mean||^2
        # The second term is constant across candidate j, so argmin reduces to a linear
        # combination of candidate_mean_sq_dist and nearest_competing_sq_dist (both already
        # available from the precomputed N×K matrices). No extra GPU kernel per cluster.
        offset = 0
        for cluster_id in range(num_clusters):
            count = counts_list[cluster_id]
            if count <= 0:
                continue
            member_indices = order[offset:offset + count]
            offset += count
            if count < refine_min_cluster_size:
                continue

            mean_sq_dist = samples_to_means_sq.index_select(0, member_indices)[:, cluster_id]

            if refine_candidate_size_mode == "adaptive_sqrt":
                cs_k = int(max(refine_candidate_size_floor,
                               min(refine_candidate_size_cap,
                                   round(math.sqrt(max(count, 1))))))
            else:
                cs_k = refine_candidate_size
            if cs_k > 0 and cs_k < count:
                cs = min(cs_k, count)
                if refine_candidate_strategy == "mean_margin" and num_centers > 1:
                    other_mask = torch.ones(num_centers, device=device, dtype=torch.bool)
                    other_mask[cluster_id] = False
                    member_to_others_sq = samples_to_centers_sq.index_select(0, member_indices)[:, other_mask]
                    nearest_sq = member_to_others_sq.min(dim=1).values.clamp_min_(1e-12)
                    margin_score = mean_sq_dist / nearest_sq
                    candidate_local = torch.topk(margin_score, k=cs, largest=False, sorted=False).indices
                elif refine_candidate_strategy == "margin_max" and num_centers > 1:
                    other_mask = torch.ones(num_centers, device=device, dtype=torch.bool)
                    other_mask[cluster_id] = False
                    member_to_others_sq = samples_to_centers_sq.index_select(0, member_indices)[:, other_mask]
                    nearest_sq = member_to_others_sq.min(dim=1).values
                    margin_score = nearest_sq - mean_sq_dist
                    candidate_local = torch.topk(margin_score, k=cs, largest=True, sorted=False).indices
                else:
                    candidate_local = torch.topk(mean_sq_dist, k=cs, largest=False, sorted=False).indices
                candidate_indices = member_indices.index_select(0, candidate_local)
                candidate_mean_sq_dist = mean_sq_dist.index_select(0, candidate_local)
            else:
                candidate_indices = member_indices
                candidate_mean_sq_dist = mean_sq_dist

            if refine_centroid_reg > 0:
                objective = (1.0 + refine_centroid_reg) * candidate_mean_sq_dist
            else:
                objective = candidate_mean_sq_dist

            if refine_separation_reg > 0 and num_centers > 1:
                other_mask = torch.ones(num_centers, device=device, dtype=torch.bool)
                other_mask[cluster_id] = False
                cand_to_others_sq = samples_to_centers_sq.index_select(0, candidate_indices)[:, other_mask]
                nearest_competing_sq = cand_to_others_sq.min(dim=1).values
                objective = objective - refine_separation_reg * nearest_competing_sq

            best_idx[cluster_id] = candidate_indices[objective.argmin()]

        return best_idx, means, counts

    # Precompute mean cluster size for adaptive sep_reg_k.
    _nonempty_counts = [c for c in counts_list if c > 0]
    _mean_size = float(sum(_nonempty_counts)) / max(len(_nonempty_counts), 1) if _nonempty_counts else 1.0

    offset = 0
    for cluster_id in range(num_clusters):
        count = counts_list[cluster_id]
        if count <= 0:
            continue
        member_indices = order[offset:offset + count]
        offset += count
        if count < refine_min_cluster_size:
            continue
        if refine_candidate_size_mode == "adaptive_sqrt":
            cs_k = int(max(refine_candidate_size_floor,
                           min(refine_candidate_size_cap,
                               round(math.sqrt(max(count, 1))))))
        else:
            cs_k = refine_candidate_size

        # Per-cluster adaptive separation_reg.
        if refine_separation_reg_mode == "adaptive_sqrt" and refine_separation_reg > 0:
            _scale = (float(count) / max(_mean_size, 1.0)) ** 0.5
            _scale = max(refine_separation_reg_scale_min, min(refine_separation_reg_scale_max, _scale))
            _sep_reg_k = refine_separation_reg * _scale
        else:
            _sep_reg_k = refine_separation_reg

        # density-adaptive sep_reg overrides all other per-cluster modes.
        if sep_reg_density_per_k is not None:
            _sep_reg_k = float(sep_reg_density_per_k[cluster_id].item())

        # Per-cluster size-adaptive strategy dispatch.
        if (refine_strategy_large_cluster and refine_strategy_size_adaptive_threshold > 0
                and count >= refine_strategy_size_adaptive_threshold):
            _effective_strategy = refine_strategy_large_cluster
        else:
            _effective_strategy = refine_candidate_strategy

        best_idx[cluster_id] = _select_bounded_l2_medoid_index_from_precomputed(
            samples=samples,
            member_indices=member_indices,
            count=count,
            cluster_id=cluster_id,
            samples_to_means_sq=samples_to_means_sq,
            samples_to_centers_sq=samples_to_centers_sq,
            candidate_size=cs_k,
            candidate_strategy=_effective_strategy,
            sample_block_size=sample_block_size,
            centroid_reg=refine_centroid_reg,
            separation_reg=_sep_reg_k,
            knn_mean_margin_pre_k_mult=knn_mean_margin_pre_k_mult,
            knn_mean_margin_score_mode=knn_mean_margin_score_mode,
            knn_mean_margin_subtractive_lambda=knn_mean_margin_subtractive_lambda,
            margin_dir_1d=margin_dir_1d,
            current_centers=current_centers,
        )
    return best_idx, means, counts


@torch.no_grad()
def initialize_medoid_indices_with_kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    args,
    cp,
    KMeans,
) -> torch.Tensor:
    """KMeans-warmstart medoid init (centroid-nearest samples).

    Alternative init modes (kmeans_l2_medoid, fps) were closed in cycles 9/29.
    """
    init_iters = min(max(args.max_iter, 1), 10)
    kmeans = build_kmeans(
        num_clusters=num_clusters,
        args=args,
        KMeans=KMeans,
        max_iter=init_iters,
    )
    labels_cp = ensure_cupy_array(kmeans.fit_predict(torch_to_cupy(samples, cp)), cp)
    centroids_cp = ensure_cupy_array(kmeans.cluster_centers_, cp)
    labels = cupy_to_torch(labels_cp).to(dtype=torch.int64, device=samples.device)
    centroids = cupy_to_torch(centroids_cp).to(dtype=torch.float32, device=samples.device)
    init_idx, _ = select_representative_indices(
        samples=samples,
        labels=labels,
        centroids=centroids,
        sample_block_size=args.sample_block_size,
    )
    return init_idx


@torch.no_grad()
def refill_empty_medoid_indices(
    min_sq_dist: torch.Tensor,
    taken_indices: torch.Tensor,
    num_fill: int,
) -> torch.Tensor:
    if num_fill <= 0:
        return torch.empty((0,), device=min_sq_dist.device, dtype=torch.int64)

    taken_mask = torch.zeros(min_sq_dist.size(0), device=min_sq_dist.device, dtype=torch.bool)
    if taken_indices.numel() > 0:
        taken_mask[taken_indices] = True

    order = torch.argsort(min_sq_dist, descending=True)
    available_order = order[~taken_mask.index_select(0, order)]
    if available_order.numel() < num_fill:
        raise RuntimeError(
            f"Unable to refill {num_fill} empty medoids from only {available_order.numel()} "
            "available samples."
        )
    return available_order[:num_fill]


@torch.no_grad()
def _compute_inertia(
    samples: torch.Tensor,
    labels: torch.Tensor,
    medoid_features: torch.Tensor,
) -> float:
    labels_flat = labels.reshape(-1).to(dtype=torch.int64, device=samples.device)
    assigned = medoid_features.index_select(0, labels_flat)
    return float((samples - assigned).square().sum().item())


@torch.no_grad()
def _apply_post_swap_polish(
    samples: torch.Tensor,
    medoid_idx: torch.Tensor,
    labels: torch.Tensor,
    args,
    top_m_override: int | None = None,
    excluded_indices: torch.Tensor | None = None,  # prevent duplicate indices
) -> tuple:
    """N-pass bounded re-selection polish on the top-M most-contested medoids.

    Supports n_passes, criterion dispatch, and
    decoupled cosine refine metric. Returns updated (medoid_idx, labels).
    """
    _top_m = top_m_override if top_m_override is not None else int(getattr(args, "kmedoid_post_swap_polish_top_m", 0))
    if _top_m <= 0 or medoid_idx.size(0) < 2:
        return medoid_idx, labels

    _n_passes = int(getattr(args, "kmedoid_post_swap_polish_n_passes", 1))
    _criterion = str(getattr(args, "kmedoid_post_swap_polish_criterion", "closest_pair"))
    _refine_metric = str(getattr(args, "kmedoid_refine_distance_metric", "l2"))

    medoid_idx = medoid_idx.clone()
    for _pass_idx in range(max(1, _n_passes)):
        _medoid_feats = samples.index_select(0, medoid_idx)
        _m = min(_top_m, medoid_idx.size(0))

        if _criterion == "cluster_size":
            _counts = torch.bincount(labels, minlength=medoid_idx.size(0)).float()
            _contested_k = torch.topk(_counts, k=_m, largest=True, sorted=False).indices
        elif _criterion == "centroid_dist":
            _K = medoid_idx.size(0)
            _centroids = torch.zeros((_K, samples.size(1)), device=samples.device, dtype=samples.dtype)
            _cnts = torch.zeros(_K, device=samples.device, dtype=torch.long)
            _centroids.index_add_(0, labels, samples)
            _cnts.index_add_(0, labels, torch.ones_like(labels))
            _centroids = _centroids / _cnts.clamp_min(1).unsqueeze(-1).to(_centroids.dtype)
            _rank_score = (_medoid_feats - _centroids).square().sum(dim=1)
            _contested_k = torch.topk(_rank_score, k=_m, largest=True, sorted=False).indices
        elif _criterion == "silhouette":
            # Silhouette-like score: (nearest_other_centroid_dist - own_centroid_dist)
            #                       / (nearest_other_centroid_dist + own_centroid_dist).
            # Low silhouette = ambiguous-boundary medoid (good polish target).
            _K = medoid_idx.size(0)
            _centroids = torch.zeros((_K, samples.size(1)), device=samples.device, dtype=samples.dtype)
            _cnts = torch.zeros(_K, device=samples.device, dtype=torch.long)
            _centroids.index_add_(0, labels, samples)
            _cnts.index_add_(0, labels, torch.ones_like(labels))
            _centroids = _centroids / _cnts.clamp_min(1).unsqueeze(-1).to(_centroids.dtype)
            _own_dist = (_medoid_feats - _centroids).norm(dim=1)
            _ctr_dist = torch.cdist(_medoid_feats, _centroids, p=2)
            _ctr_dist.fill_diagonal_(float("inf"))
            _other_dist = _ctr_dist.min(dim=1).values
            _silhouette = (_other_dist - _own_dist) / (_other_dist + _own_dist).clamp_min(1e-8)
            _contested_k = torch.topk(_silhouette, k=_m, largest=False, sorted=False).indices
        elif _criterion == "within_variance":
            # Within-cluster variance per cluster. HIGH variance = dispersed cluster (polish target).
            _K = medoid_idx.size(0)
            _diff_sq = samples.sub(_medoid_feats[labels]).square().sum(dim=1)
            _withinvar = torch.zeros(_K, device=samples.device, dtype=samples.dtype)
            _withinvar.scatter_add_(0, labels, _diff_sq)
            _cnts2 = torch.zeros(_K, device=samples.device, dtype=torch.long)
            _cnts2.scatter_add_(0, labels, torch.ones_like(labels))
            _withinvar = _withinvar / _cnts2.clamp_min(1).to(_withinvar.dtype)
            _contested_k = torch.topk(_withinvar, k=_m, largest=True, sorted=False).indices
        else:  # closest_pair (c24 default)
            _md_dist = torch.cdist(_medoid_feats, _medoid_feats, p=2)
            _md_dist.fill_diagonal_(float("inf"))
            _min_dist = _md_dist.min(dim=1).values
            _contested_k = torch.topk(_min_dist, k=_m, largest=False, sorted=False).indices

        _fwd_select_k = int(getattr(args, "kmedoid_post_swap_polish_forward_select_k", 1))
        _margin_dir_1d = bool(int(getattr(args, "kmedoid_refine_margin_dir_1d", 0)))
        _cs_polish = int(getattr(args, "kmedoid_refine_candidate_size", 0))
        for _k in _contested_k.tolist():
            _member_mask = (labels == _k)
            _member_indices = _member_mask.nonzero(as_tuple=False).squeeze(-1)
            # exclude sample indices already used by FPS fill (prevents dup output).
            if excluded_indices is not None and excluded_indices.numel() > 0:
                _keep_mask = ~torch.isin(_member_indices, excluded_indices)
                _member_indices = _member_indices[_keep_mask]
            if _member_indices.numel() <= 1:
                continue
            _mean_k = samples.index_select(0, _member_indices).mean(dim=0)
            if _fwd_select_k > 1 and _cs_polish > 0:
                # forward-selection K=_fwd_select_k. Take top-K nearest-to-mean
                # candidates, then pick the one that maximizes min pairwise distance to
                # other medoids (forward-select on closest-pair criterion).
                _member_feats_pol = samples.index_select(0, _member_indices)
                _own_sq_pol = (_member_feats_pol - _mean_k.view(1, -1)).square().sum(dim=1)
                _k_take = min(_fwd_select_k, _member_indices.numel())
                _topk_local_pol = torch.topk(_own_sq_pol, k=_k_take, largest=False, sorted=False).indices
                _cand_idx_pol = _member_indices.index_select(0, _topk_local_pol)
                _other_mask_pol = torch.arange(_medoid_feats.size(0), device=samples.device) != _k
                _other_medoids_pol = _medoid_feats[_other_mask_pol]
                _cand_feats_pol = samples.index_select(0, _cand_idx_pol)
                if _other_medoids_pol.size(0) > 0:
                    _d_co_pol = torch.cdist(_cand_feats_pol, _other_medoids_pol, p=2)
                    _min_d_pol = _d_co_pol.min(dim=1).values
                    _best_pol = int(_min_d_pol.argmax().item())
                    _new_idx = _cand_idx_pol[_best_pol]
                else:
                    _new_idx = _cand_idx_pol[0]
            else:
                _new_idx = select_bounded_l2_medoid_index(
                    samples=samples,
                    member_indices=_member_indices,
                    mean=_mean_k,
                    candidate_size=_cs_polish,
                    candidate_strategy=str(getattr(args, "kmedoid_refine_candidate_strategy", "mean")),
                    current_centers=_medoid_feats,
                    cluster_id=_k,
                    sample_block_size=args.sample_block_size,
                    centroid_reg=float(getattr(args, "kmedoid_refine_centroid_reg", 0.0)),
                    separation_reg=float(getattr(args, "kmedoid_refine_separation_reg", 0.0)),
                    knn_mean_margin_pre_k_mult=float(getattr(args, "knn_mean_margin_pre_k_mult", 4.0)),
                    knn_mean_margin_score_mode=str(getattr(args, "knn_mean_margin_score_mode", "ratio")),
                    knn_mean_margin_subtractive_lambda=float(getattr(args, "knn_mean_margin_subtractive_lambda", 1.0)),
                    refine_distance_metric=_refine_metric,
                    margin_dir_1d=_margin_dir_1d,
                )
            medoid_idx[_k] = _new_idx

        medoid_feats_new = samples.index_select(0, medoid_idx)
        labels, _ = assign_to_centers(
            samples=samples,
            centers=medoid_feats_new,
            sample_block_size=args.sample_block_size,
        )

    return medoid_idx, labels


@torch.no_grad()
def run_kmedoid_multi_start(
    samples: torch.Tensor,
    num_clusters: int,
    args,
    cp,
    KMeans,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run k-medoid n_init times; select best by criterion ∈ {inertia, max_dispersion}."""
    n_init = int(getattr(args, "kmedoid_n_init", 1))
    if n_init <= 1:
        return run_kmedoid(samples, num_clusters, args, cp, KMeans)

    _criterion = str(getattr(args, "kmedoid_multistart_criterion", "inertia"))
    best_score = float("inf") if _criterion == "inertia" else float("-inf")
    best_result = None
    original_state = args.random_state
    for seed_idx in range(n_init):
        args.random_state = original_state + seed_idx * 997
        medoid_idx, labels, medoid_features, iters = run_kmedoid(samples, num_clusters, args, cp, KMeans)
        if _criterion == "max_dispersion":
            # Score = minimum pairwise medoid distance (larger = more dispersed = better coverage).
            _md = torch.cdist(medoid_features, medoid_features, p=2)
            _md.fill_diagonal_(float("inf"))
            score = float(_md.min().item())
            if score > best_score:
                best_score = score
                best_result = (medoid_idx, labels, medoid_features, iters)
        else:  # inertia
            score = _compute_inertia(samples, labels, medoid_features)
            if score < best_score:
                best_score = score
                best_result = (medoid_idx, labels, medoid_features, iters)
    args.random_state = original_state
    return best_result


@torch.no_grad()
def run_kmedoid(
    samples: torch.Tensor,
    num_clusters: int,
    args,
    cp,
    KMeans,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    _init_mode = str(getattr(args, "kmedoid_init_mode", "random"))
    if _init_mode == "fps":
        # Farthest-point sampling init: K maximally-dispersed initial medoids.
        K = num_clusters
        N = samples.size(0)
        init_indices = torch.zeros(K, dtype=torch.long, device=samples.device)
        _seed_idx = int(torch.randint(0, N, (1,), device=samples.device).item())
        init_indices[0] = _seed_idx
        _min_sq_dist = torch.full((N,), float("inf"), device=samples.device, dtype=torch.float32)
        _picked_feat = samples[_seed_idx].view(1, -1)
        for _k in range(1, K):
            _new_sq = (samples - _picked_feat).square().sum(dim=1)
            _min_sq_dist = torch.minimum(_min_sq_dist, _new_sq)
            _next = int(_min_sq_dist.argmax().item())
            init_indices[_k] = _next
            _picked_feat = samples[_next].view(1, -1)
        medoid_idx = init_indices
    else:
        medoid_idx = initialize_medoid_indices_with_kmeans(
            samples=samples,
            num_clusters=num_clusters,
            args=args,
            cp=cp,
            KMeans=KMeans,
        )

    if medoid_idx.numel() != num_clusters:
        raise RuntimeError(
            f"k_medoid init returned {medoid_idx.numel()} medoids for requested K={num_clusters}."
        )

    _refine_two_stage = bool(getattr(args, "kmedoid_refine_two_stage", 0))
    _margin_dir_1d = bool(int(getattr(args, "kmedoid_refine_margin_dir_1d", 0)))
    _sep_reg_density = bool(int(getattr(args, "kmedoid_refine_sep_reg_density", 0)))
    iterations = 0
    for iteration in range(args.max_iter):
        medoid_features = samples.index_select(0, medoid_idx)
        labels, min_sq_dist = assign_to_centers(
            samples=samples,
            centers=medoid_features,
            sample_block_size=args.sample_block_size,
        )
        # density-adaptive per-cluster sep_reg.
        if _sep_reg_density:
            _sep_low = float(getattr(args, "kmedoid_refine_sep_reg_low", 0.15))
            _sep_high = float(getattr(args, "kmedoid_refine_sep_reg_high", 0.28))
            _K = num_clusters
            _own_dists = (samples - medoid_features[labels]).norm(dim=1)
            _mean_intra = torch.zeros(_K, device=samples.device, dtype=samples.dtype)
            _cnts_dens = torch.zeros(_K, device=samples.device, dtype=torch.long)
            _mean_intra.scatter_add_(0, labels, _own_dists)
            _cnts_dens.scatter_add_(0, labels, torch.ones_like(labels))
            _mean_intra = _mean_intra / _cnts_dens.clamp_min(1).to(_mean_intra.dtype)
            _density = 1.0 / _mean_intra.clamp_min(1e-8)
            _rank = torch.argsort(torch.argsort(_density)).to(samples.dtype) / max(1.0, float(_K - 1))
            sep_reg_per_k = _sep_low + (_sep_high - _sep_low) * _rank
        else:
            sep_reg_per_k = None
        _effective_strategy = (
            "mean_margin" if (_refine_two_stage and iteration == 0)
            else args.kmedoid_refine_candidate_strategy
        )
        new_medoid_idx, means, counts = update_medoid_indices_from_assignments(
            samples=samples,
            labels=labels,
            current_centers=medoid_features,
            num_clusters=num_clusters,
            sample_block_size=args.sample_block_size,
            refine_candidate_size=args.kmedoid_refine_candidate_size,
            refine_candidate_strategy=_effective_strategy,
            refine_min_cluster_size=args.kmedoid_refine_min_cluster_size,
            refine_centroid_reg=args.kmedoid_refine_centroid_reg,
            refine_separation_reg=args.kmedoid_refine_separation_reg,
            cdist_impl=getattr(args, "cdist_impl", "torch"),
            cdist_fp16=bool(getattr(args, "cdist_fp16", False)),
            cdist_compile=bool(getattr(args, "cdist_compile", False)),
            refine_batched=bool(getattr(args, "refine_batched", False)),
            refine_candidate_size_mode=getattr(args, "kmedoid_refine_candidate_size_mode", "fixed"),
            refine_candidate_size_floor=int(getattr(args, "kmedoid_refine_candidate_size_floor", 4)),
            refine_candidate_size_cap=int(getattr(args, "kmedoid_refine_candidate_size_cap", 16)),
            knn_mean_margin_pre_k_mult=float(getattr(args, "knn_mean_margin_pre_k_mult", 4.0)),
            knn_mean_margin_score_mode=str(getattr(args, "knn_mean_margin_score_mode", "ratio")),
            knn_mean_margin_subtractive_lambda=float(getattr(args, "knn_mean_margin_subtractive_lambda", 1.0)),
            refine_separation_reg_mode=str(getattr(args, "kmedoid_refine_separation_reg_mode", "fixed")),
            refine_separation_reg_scale_min=float(getattr(args, "kmedoid_refine_separation_reg_scale_min", 0.5)),
            refine_separation_reg_scale_max=float(getattr(args, "kmedoid_refine_separation_reg_scale_max", 2.0)),
            refine_strategy_large_cluster=str(getattr(args, "kmedoid_refine_strategy_large_cluster", "") or ""),
            refine_strategy_size_adaptive_threshold=int(getattr(args, "kmedoid_refine_strategy_size_adaptive_threshold", 0)),
            margin_dir_1d=_margin_dir_1d,
            sep_reg_density_per_k=sep_reg_per_k,
        )

        empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten()
        if empty_clusters.numel() > 0:
            nonempty_clusters = (counts > 0).nonzero(as_tuple=False).flatten()
            taken = new_medoid_idx.index_select(0, nonempty_clusters)
            refill = refill_empty_medoid_indices(
                min_sq_dist=min_sq_dist,
                taken_indices=taken,
                num_fill=int(empty_clusters.numel()),
            )
            new_medoid_idx.index_copy_(0, empty_clusters, refill)
            means.index_copy_(0, empty_clusters, samples.index_select(0, refill))

        iterations = iteration + 1
        if torch.equal(new_medoid_idx, medoid_idx):
            medoid_idx = new_medoid_idx
            break
        medoid_idx = new_medoid_idx

    medoid_features = samples.index_select(0, medoid_idx)
    labels, _ = assign_to_centers(
        samples=samples,
        centers=medoid_features,
        sample_block_size=args.sample_block_size,
    )

    # Post-convergence polish: N-pass bounded re-selection on top-M contested medoids.
    medoid_idx, labels = _apply_post_swap_polish(
        samples=samples, medoid_idx=medoid_idx, labels=labels, args=args,
    )
    medoid_features = samples.index_select(0, medoid_idx)

    return medoid_idx, labels, medoid_features, iterations
