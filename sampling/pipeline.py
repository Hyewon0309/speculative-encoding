from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch

_PCA_PROJECTION_CACHE: Dict[Tuple, torch.Tensor] = {}


def _get_pca_projection(samples: torch.Tensor, d_new: int, device: torch.device) -> torch.Tensor:
    """Compute (or fetch from cache) a D_old×d_new PCA projection matrix.

    Bootstrapped from the first slide that calls this function; cached for all
    subsequent slides in the same process. PCA cost is inside the T_v fence.
    """
    key = (int(samples.size(1)), int(d_new), str(device))
    if key in _PCA_PROJECTION_CACHE:
        return _PCA_PROJECTION_CACHE[key]
    centered = samples - samples.mean(dim=0, keepdim=True)
    # torch.pca_lowrank returns U (N, q), S (q,), V (D, q); samples @ V projects to (N, q).
    _, _, V = torch.pca_lowrank(centered, q=d_new, niter=2)
    _PCA_PROJECTION_CACHE[key] = V.detach()
    return V

from .cli import parse_args, validate_args
from .clustering import (
    _apply_post_swap_polish,
    resolve_effective_ratio,
    resolve_num_clusters,
    resolve_target_selected,
    run_kmedoid,
    run_kmedoid_multi_start,
    select_representative_indices,
)
from .io import list_pt_files, load_feature_tensor, validate_flat_output_names
from .rapids import build_kmeans, cupy_to_torch, ensure_cupy_array, import_rapids, synchronize_if_needed, torch_to_cupy
from .selection import (
    knn_swap_refine,
    sample_cluster_additions,
    sample_cluster_additions_to_budget,
    swap_refine_margin_per_cluster,
)


def process_one_file(path, output_path, args, cp, KMeans, device: torch.device):
    t0 = time.perf_counter()
    samples_cpu = load_feature_tensor(path, use_mmap=args.mmap)
    num_samples, feat_dim = samples_cpu.shape
    effective_ratio = resolve_effective_ratio(
        num_samples=num_samples,
        base_ratio=args.ratio,
        adaptive_ratio_schedule=args.adaptive_ratio_schedule,
    )
    target_selected = resolve_target_selected(
        num_samples=num_samples,
        budget_ratio=args.budget_ratio,
    )
    num_clusters = resolve_num_clusters(
        num_samples=num_samples,
        ratio=effective_ratio,
        min_clusters=args.min_clusters,
        target_selected=target_selected,
    )

    synchronize_if_needed(device)
    t_load_done = time.perf_counter()

    samples = samples_cpu.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()
    del samples_cpu

    t_sampler_start = time.perf_counter()
    _rp_dim = int(getattr(args, 'feature_random_projection_dim', 0))
    if _rp_dim > 0 and _rp_dim < samples.size(1):
        g = torch.Generator(device=device)
        g.manual_seed(42)
        D_old = samples.size(1)
        R = torch.randn(D_old, _rp_dim, generator=g, device=device, dtype=samples.dtype) / (_rp_dim ** 0.5)
        samples = (samples @ R).contiguous()
    _pca_dim = int(getattr(args, 'feature_pca_projection_dim', 0))
    if _pca_dim > 0 and _pca_dim < samples.size(1):
        V = _get_pca_projection(samples, _pca_dim, device)
        samples = ((samples - samples.mean(dim=0, keepdim=True)) @ V).contiguous()

    clustering_iters = 0
    if args.clustering == "k_means":
        samples_cp = torch_to_cupy(samples, cp)
        kmeans = build_kmeans(num_clusters=num_clusters, args=args, KMeans=KMeans)
        labels_cp = ensure_cupy_array(kmeans.fit_predict(samples_cp), cp)
        centroids_cp = ensure_cupy_array(kmeans.cluster_centers_, cp)

        labels = cupy_to_torch(labels_cp).to(dtype=torch.int64, device=device)
        centroids = cupy_to_torch(centroids_cp).to(dtype=torch.float32, device=device)
        clustering_iters = int(getattr(kmeans, "n_iter_", 0) or 0)

        rep_idx, used_clusters = select_representative_indices(
            samples=samples,
            labels=labels,
            centroids=centroids,
            sample_block_size=args.sample_block_size,
        )
    elif args.clustering == "kcenter_greedy":
        # Pure k-Center Greedy baseline: greedy farthest-first traversal on the
        # full feature set until target_selected patches are chosen. No
        # clustering, no entropy-budget allocation, no margin-swap refinement.
        if target_selected is None:
            raise ValueError(
                "--clustering=kcenter_greedy requires --budget_ratio so that "
                "target_selected is well-defined."
            )
        from .selection import sample_cluster_fps as _kcg_fps
        _global_centroid = samples.mean(dim=0)
        _all_indices = torch.arange(num_samples, device=device, dtype=torch.int64)
        _empty_seed = torch.empty((0,), device=device, dtype=torch.int64)
        rep_idx = _kcg_fps(
            samples=samples,
            centroid=_global_centroid,
            available_indices=_all_indices,
            seed_indices=_empty_seed,
            num_pick=int(target_selected),
            fps_metric=args.fps_metric,
            fps_l2_normalize=bool(getattr(args, "fps_l2_normalize", False)),
            fps_hybrid_alpha=float(getattr(args, "fps_hybrid_alpha", 1.0)),
            fps_magnitude_beta=float(getattr(args, "fps_magnitude_beta", 0.0)),
            fps_aggregate_mode=str(getattr(args, "fps_aggregate_mode", "min")),
        )
        # Synthesize trivial single-cluster bookkeeping so downstream stats /
        # logging code does not see undefined `labels` / `centroids`.
        labels = torch.zeros(num_samples, dtype=torch.int64, device=device)
        centroids = _global_centroid.unsqueeze(0)
        used_clusters = 1
    else:
        rep_idx, labels, centroids, clustering_iters = run_kmedoid_multi_start(
            samples=samples,
            num_clusters=num_clusters,
            args=args,
            cp=cp,
            KMeans=KMeans,
        )
        used_clusters = int(torch.bincount(labels, minlength=num_clusters).gt(0).sum().item())
        _k_medoid_pre_fps = rep_idx.clone()  # K medoid indices before FPS fill (for post-fvec polish)

    base_selected = int(rep_idx.numel())

    # k-Center Greedy already produced the full target_selected output; skip
    # the cluster-extras / refinement / polish stages entirely.
    _kcg_skip_extras = (args.clustering == "kcenter_greedy")

    if _kcg_skip_extras:
        extra_idx = torch.empty((0,), device=device, dtype=torch.int64)
    elif target_selected is not None:
        if target_selected < base_selected:
            raise ValueError(
                f"Target selected count {target_selected} is smaller than base cluster "
                f"representative count {base_selected}. Lower --ratio or raise --budget_ratio."
            )
        extra_idx = sample_cluster_additions_to_budget(
            samples=samples,
            labels=labels,
            centroids=centroids,
            base_indices=rep_idx,
            target_selected=target_selected,
            add_sample=args.add_sample,
            budget_allocation=args.budget_allocation,
            fps_seed_global_cap=int(args.fps_seed_global_cap),
            fps_metric=args.fps_metric,
            entropy_temperature=args.budget_allocation_entropy_temperature,
            fps_l2_normalize=bool(getattr(args, "fps_l2_normalize", False)),
            fps_hybrid_alpha=float(getattr(args, "fps_hybrid_alpha", 1.0)),
            fps_magnitude_beta=float(getattr(args, "fps_magnitude_beta", 0.0)),
            fps_aggregate_mode=str(getattr(args, "fps_aggregate_mode", "min")),
        )
    else:
        extra_idx = sample_cluster_additions(
            samples=samples,
            labels=labels,
            centroids=centroids,
            base_indices=rep_idx,
            add_sample=args.add_sample,
            add_sample_num=args.add_sample_num,
            fps_metric=args.fps_metric,
            fps_l2_normalize=bool(getattr(args, "fps_l2_normalize", False)),
            fps_hybrid_alpha=float(getattr(args, "fps_hybrid_alpha", 1.0)),
        )
    if extra_idx.numel() > 0:
        rep_idx = torch.cat((rep_idx, extra_idx), dim=0)

    _n_passes = int(args.margin_swap_refine_passes)
    _post_spread = int(getattr(args, "post_antims_max_spread_k", 0))
    if _n_passes > 0 or _post_spread > 0:
        rep_idx = swap_refine_margin_per_cluster(
            samples=samples,
            labels=labels,
            selected_indices=rep_idx,
            n_passes=_n_passes,
            objective_sign=int(args.margin_swap_objective_sign),
            margin_swap_min_cluster_size=int(getattr(args, "margin_swap_min_cluster_size", 2)),
            margin_swap_cross_cluster_top_k=int(getattr(args, "margin_swap_cross_cluster_top_k", 0)),
            margin_swap_vectorize_global_cdist=bool(getattr(args, "margin_swap_vectorize_global_cdist", False)),
            margin_swap_top_m_unsel=int(getattr(args, "margin_swap_top_m_unsel", 0)),
            margin_swap_vectorize_mode=str(getattr(args, "margin_swap_vectorize_mode", "A")),
            margin_swap_variance_skip_threshold=float(getattr(args, "margin_swap_variance_skip_threshold", 0.0)),
            adaptive_n=bool(getattr(args, "margin_swap_adaptive_n", 0)),
            other_sel_subsample=int(getattr(args, "margin_swap_other_sel_subsample", 0)),
            margin_agg=str(getattr(args, "margin_swap_margin_agg", "mean")),
            margin_swap_other_sel_approx=str(getattr(args, "margin_swap_other_sel_approx", "none")),
            margin_swap_batch_k=int(getattr(args, "margin_swap_batch_k", 1)),
            margin_swap_chain_k=int(getattr(args, "margin_swap_chain_k", 1)),
            post_antims_max_spread_k=_post_spread,
            profile_anti_ms=bool(getattr(args, "profile_anti_ms", 0)),
            margin_swap_vectorize_full_kloop=int(getattr(args, "margin_swap_vectorize_full_kloop", 0)),
            margin_swap_jit_script=bool(getattr(args, "margin_swap_jit_script", 0)),
            margin_swap_agg_weight=str(getattr(args, "margin_swap_agg_weight", "uniform")),
            margin_swap_chain_k_top_pct=float(getattr(args, "margin_swap_chain_k_top_pct", 0.0)),
            margin_swap_post_fvec_seq_top_pct=float(getattr(args, "margin_swap_post_fvec_seq_top_pct", 0.0)),
            margin_swap_post_fvec_seq_mode=str(getattr(args, "margin_swap_post_fvec_seq_mode", "single")),
            margin_swap_post_fvec_seq_pass2_top_pct=float(getattr(args, "margin_swap_post_fvec_seq_pass2_top_pct", 0.0)),
            margin_swap_post_fvec_seq_pass2_mode=str(getattr(args, "margin_swap_post_fvec_seq_pass2_mode", "stale")),
            margin_swap_post_fvec_seq_chain_k=int(getattr(args, "margin_swap_post_fvec_seq_chain_k", 2)),
            margin_swap_num_candidates=int(getattr(args, "margin_swap_num_candidates", 1)),
            margin_swap_torchcompile_mode=str(getattr(args, "margin_swap_torchcompile_mode", "none")),
            margin_swap_fvec_microbatch_b=int(getattr(args, "margin_swap_fvec_microbatch_b", 0)),
            margin_swap_inter_cluster_spread_weight=float(getattr(args, "margin_swap_inter_cluster_spread_weight", 0.0)),
            margin_swap_adaptive_np_large_threshold=int(getattr(args, "margin_swap_adaptive_np_large_threshold", 0)),
            margin_swap_post_fvec_seq_select_mode=str(getattr(args, "margin_swap_post_fvec_seq_select_mode", "delta_rank")),
        )

    # post-polish pfseq re-run (fresh fvec+pfseq after polish, with given top_pct).
    _polish_pfseq_pct = float(getattr(args, "kmedoid_polish_pfseq_rerun_pct", 0.0))
    _polish_top_m = int(getattr(args, "kmedoid_post_swap_polish_top_m", 0))
    if _polish_pfseq_pct > 0.0 and _polish_top_m > 0 and _n_passes > 0:
        rep_idx = swap_refine_margin_per_cluster(
            samples=samples,
            labels=labels,
            selected_indices=rep_idx,
            n_passes=1,
            objective_sign=int(args.margin_swap_objective_sign),
            margin_swap_min_cluster_size=int(getattr(args, "margin_swap_min_cluster_size", 2)),
            margin_swap_cross_cluster_top_k=0,
            margin_swap_vectorize_global_cdist=False,
            margin_swap_top_m_unsel=0,
            margin_swap_vectorize_mode="A",
            margin_swap_variance_skip_threshold=0.0,
            adaptive_n=False,
            other_sel_subsample=0,
            margin_agg="mean",
            margin_swap_other_sel_approx="none",
            margin_swap_batch_k=1,
            margin_swap_chain_k=int(getattr(args, "margin_swap_chain_k", 1)),
            post_antims_max_spread_k=0,
            profile_anti_ms=False,
            margin_swap_vectorize_full_kloop=1,
            margin_swap_jit_script=False,
            margin_swap_agg_weight="uniform",
            margin_swap_chain_k_top_pct=0.0,
            margin_swap_post_fvec_seq_top_pct=_polish_pfseq_pct,
            margin_swap_post_fvec_seq_mode=str(getattr(args, "margin_swap_post_fvec_seq_mode", "chain_k")),
            margin_swap_post_fvec_seq_pass2_top_pct=0.0,
            margin_swap_post_fvec_seq_pass2_mode="stale",
            margin_swap_post_fvec_seq_chain_k=int(getattr(args, "margin_swap_post_fvec_seq_chain_k", 2)),
            margin_swap_num_candidates=1,
            margin_swap_torchcompile_mode="none",
            margin_swap_fvec_microbatch_b=0,
            margin_swap_inter_cluster_spread_weight=0.0,
            margin_swap_adaptive_np_large_threshold=0,
            margin_swap_post_fvec_seq_select_mode="delta_rank",
        )

    # post-fvec polish with relabel fix — recompute labels from post-fvec medoids before polish.
    _post_fvec_top_m = int(getattr(args, "kmedoid_post_fvec_polish_top_m", 0))
    _post_fvec_relabel = bool(int(getattr(args, "kmedoid_post_fvec_polish_relabel", 0)))
    if _post_fvec_top_m > 0 and args.clustering == "k_medoid":
        _post_fvec_medoids = rep_idx[:num_clusters].clone()
        _post_fvec_extras = rep_idx[num_clusters:].clone() if rep_idx.numel() > num_clusters else rep_idx.new_empty((0,), dtype=rep_idx.dtype)
        # recompute labels from post-fvec medoids so polish sees current cluster membership.
        if _post_fvec_relabel:
            from .clustering import assign_to_centers
            _post_fvec_medoid_feats = samples.index_select(0, _post_fvec_medoids)
            labels, _ = assign_to_centers(
                samples=samples,
                centers=_post_fvec_medoid_feats,
                sample_block_size=args.sample_block_size,
            )
        _new_medoids, labels = _apply_post_swap_polish(
            samples=samples,
            medoid_idx=_post_fvec_medoids,
            labels=labels,
            args=args,
            top_m_override=_post_fvec_top_m,
            excluded_indices=_post_fvec_extras,
        )
        # defensive dedup — if any polish-picked medoid collides with an existing rep_idx
        # entry (should not happen post-relabel, but guard for overlap edge cases), replace with
        # original pre-polish medoid.
        if _post_fvec_relabel:
            _combined = torch.cat((_new_medoids, _post_fvec_extras)) if _post_fvec_extras.numel() > 0 else _new_medoids
            _unique_count = int(_combined.unique().numel())
            if _unique_count != int(_combined.numel()):
                _taken = set()
                _fixed = _new_medoids.clone()
                if _post_fvec_extras.numel() > 0:
                    _taken.update(int(x) for x in _post_fvec_extras.tolist())
                for _i, _m in enumerate(_new_medoids.tolist()):
                    if _m in _taken:
                        _fallback = int(_post_fvec_medoids[_i].item())
                        if _fallback not in _taken:
                            _fixed[_i] = _post_fvec_medoids[_i]
                            _taken.add(_fallback)
                    else:
                        _taken.add(_m)
                _new_medoids = _fixed
        if _post_fvec_extras.numel() > 0:
            rep_idx = torch.cat((_new_medoids, _post_fvec_extras), dim=0)
        else:
            rep_idx = _new_medoids

    # FPS-extras polish — cluster-local re-selection for each extra, targeting
    # within-cluster points that maximize min-distance to the full current rep_idx set.
    _fps_extras_polish_enable = bool(int(getattr(args, "fps_extras_polish_enable", 0)))
    if _fps_extras_polish_enable and rep_idx.numel() > num_clusters:
        _K = num_clusters
        _medoids_cur = rep_idx[:_K]
        _extras_cur = rep_idx[_K:].clone()
        _medoid_feats = samples.index_select(0, _medoids_cur)
        _extras_feats = samples.index_select(0, _extras_cur)
        _d_em = torch.cdist(_extras_feats, _medoid_feats, p=2)
        _extras_home = _d_em.argmin(dim=1)
        for _i in range(_extras_cur.size(0)):
            _home_k = int(_extras_home[_i].item())
            _member_mask = (labels == _home_k)
            _member_indices = _member_mask.nonzero(as_tuple=False).squeeze(-1)
            if _member_indices.numel() <= 1:
                continue
            _rep_now = torch.cat((_medoids_cur, _extras_cur), dim=0)
            _in_rep = torch.isin(_member_indices, _rep_now)
            _candidate_pool = _member_indices[~_in_rep]
            if _candidate_pool.numel() == 0:
                continue
            _cand_feats = samples.index_select(0, _candidate_pool)
            _rep_feats = samples.index_select(0, _rep_now)
            _d_cr = torch.cdist(_cand_feats, _rep_feats, p=2)
            _min_d = _d_cr.min(dim=1).values
            _best_local = int(_min_d.argmax().item())
            _extras_cur[_i] = _candidate_pool[_best_local]
        rep_idx = torch.cat((_medoids_cur, _extras_cur), dim=0)

    # re-FPS extras after refine — discard pre-fvec extras, re-greedy-FPS using
    # post-fvec medoids as seed. Labels must be current (relabeled if Change 1 flag is on).
    _fps_extras_reseed = bool(int(getattr(args, "fps_extras_reseed_after_refine", 0)))
    if _fps_extras_reseed and rep_idx.numel() > num_clusters and target_selected is not None:
        _K = num_clusters
        _medoids_cur = rep_idx[:_K].clone()
        from .clustering import assign_to_centers
        _med_feats = samples.index_select(0, _medoids_cur)
        labels_reseed, _ = assign_to_centers(
            samples=samples,
            centers=_med_feats,
            sample_block_size=args.sample_block_size,
        )
        _centroids_re = torch.zeros((_K, samples.size(1)), device=samples.device, dtype=samples.dtype)
        _cnts_re = torch.zeros(_K, device=samples.device, dtype=torch.long)
        _centroids_re.index_add_(0, labels_reseed, samples)
        _cnts_re.index_add_(0, labels_reseed, torch.ones_like(labels_reseed))
        _centroids_re = _centroids_re / _cnts_re.clamp_min(1).unsqueeze(-1).to(_centroids_re.dtype)
        _new_extras = sample_cluster_additions_to_budget(
            samples=samples,
            labels=labels_reseed,
            centroids=_centroids_re,
            base_indices=_medoids_cur,
            target_selected=int(target_selected),
            add_sample=args.add_sample,
            budget_allocation=args.budget_allocation,
            fps_seed_global_cap=int(args.fps_seed_global_cap),
            fps_metric=args.fps_metric,
            entropy_temperature=args.budget_allocation_entropy_temperature,
            fps_l2_normalize=bool(getattr(args, "fps_l2_normalize", False)),
            fps_hybrid_alpha=float(getattr(args, "fps_hybrid_alpha", 1.0)),
            fps_magnitude_beta=float(getattr(args, "fps_magnitude_beta", 0.0)),
            fps_aggregate_mode=str(getattr(args, "fps_aggregate_mode", "min")),
        )
        rep_idx = torch.cat((_medoids_cur, _new_extras), dim=0)

    if getattr(args, "knn_swap_refine_k", 0) > 0:
        rep_idx = knn_swap_refine(
            samples=samples,
            labels=labels,
            selected_indices=rep_idx,
            k_nn=int(args.knn_swap_refine_k),
            objective_sign=int(args.margin_swap_objective_sign),
        )

    unique_indices = int(rep_idx.unique().numel())
    duplicate_indices = int(rep_idx.numel() - unique_indices)
    if args.sort_indices:
        rep_idx = torch.sort(rep_idx).values

    synchronize_if_needed(device)
    t_sampler_end = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.time_only:
        np.save(output_path, rep_idx.cpu().numpy().astype(np.int64, copy=False), allow_pickle=False)

    synchronize_if_needed(device)
    elapsed = time.perf_counter() - t0
    load_elapsed = t_load_done - t0
    sampler_elapsed = t_sampler_end - t_sampler_start
    print(f"[gpu-time] budget={args.budget_ratio if args.budget_ratio is not None else args.ratio:.6f} file={path.stem} sampler_sec={sampler_elapsed:.4f}")
    return {
        "num_samples": num_samples,
        "feat_dim": feat_dim,
        "num_clusters": num_clusters,
        "effective_ratio": float(effective_ratio),
        "target_selected": int(target_selected) if target_selected is not None else None,
        "min_clusters": int(args.min_clusters),
        "used_clusters": used_clusters,
        "clustering": args.clustering,
        "clustering_iters": int(clustering_iters),
        "base_selected": base_selected,
        "extra_selected": int(extra_idx.numel()),
        "total_selected": int(rep_idx.numel()),
        "unique_indices": unique_indices,
        "duplicate_indices": duplicate_indices,
        "sorted_indices": bool(args.sort_indices),
        "add_sample": args.add_sample,
        "add_sample_num": int(args.add_sample_num),
        "fps_seed_global_cap": int(args.fps_seed_global_cap),
        "fps_metric": args.fps_metric,
        "budget_allocation": args.budget_allocation,
        "budget_allocation_entropy_temperature": float(args.budget_allocation_entropy_temperature),
        "time_only": bool(args.time_only),
        "kmedoid_refine_candidate_size": int(args.kmedoid_refine_candidate_size),
        "kmedoid_refine_candidate_strategy": args.kmedoid_refine_candidate_strategy,
        "kmedoid_refine_min_cluster_size": int(args.kmedoid_refine_min_cluster_size),
        "kmedoid_refine_centroid_reg": float(args.kmedoid_refine_centroid_reg),
        "kmedoid_refine_separation_reg": float(args.kmedoid_refine_separation_reg),
        "margin_swap_refine_passes": int(args.margin_swap_refine_passes),
        "margin_swap_objective_sign": int(args.margin_swap_objective_sign),
        "load_sec": load_elapsed,
        "sampler_sec": sampler_elapsed,
        "total_sec": elapsed,
    }


def main():
    args = parse_args()
    validate_args(args)

    cp, KMeans = import_rapids()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"RAPIDS K-means requires CUDA, got device={args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.cuda.set_device(device)
    cp.cuda.Device(device.index or 0).use()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    files = list_pt_files(args.input_dir, recursive=args.recursive)
    validate_flat_output_names(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    start_time = time.perf_counter()

    with torch.inference_mode():
        for idx, path in enumerate(files, start=1):
            output_path = args.output_dir / f"{path.stem}.npy"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{idx}/{len(files)}] skip {path.name} -> {output_path.name} (exists)")
                continue

            stats = process_one_file(
                path=path,
                output_path=output_path,
                args=args,
                cp=cp,
                KMeans=KMeans,
                device=device,
            )
            processed += 1
            print(
                f"[{idx}/{len(files)}] {path.name} "
                f"N={stats['num_samples']:,} D={stats['feat_dim']} "
                f"K={stats['num_clusters']:,} ratio={stats['effective_ratio']:.6f} "
                f"used={stats['used_clusters']:,} "
                f"target={stats['target_selected'] if stats['target_selected'] is not None else '-'} "
                f"cluster={stats['clustering']} iter={stats['clustering_iters']:,} "
                f"base={stats['base_selected']:,} extra={stats['extra_selected']:,} "
                f"total={stats['total_selected']:,} "
                f"unique={stats['unique_indices']:,} dup={stats['duplicate_indices']:,} "
                f"add={stats['add_sample']}x{stats['add_sample_num']} cap={stats['fps_seed_global_cap']} "
                f"budget_alloc={stats['budget_allocation']} "
                f"refine={stats['kmedoid_refine_candidate_size']:,}/{stats['kmedoid_refine_candidate_strategy']} "
                f"sep_reg={stats['kmedoid_refine_separation_reg']:.3f} "
                f"ms_passes={stats['margin_swap_refine_passes']}@{stats['margin_swap_objective_sign']:+d} "
                f"load={stats['load_sec']:.2f}s sampler={stats['sampler_sec']:.2f}s total={stats['total_sec']:.2f}s"
            )

    total_elapsed = time.perf_counter() - start_time
    print(
        f"Finished. processed={processed} skipped={skipped} "
        f"elapsed={total_elapsed:.2f}s output_dir={args.output_dir}"
    )
    return 0
