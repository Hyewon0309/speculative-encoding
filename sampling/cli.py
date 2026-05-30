from __future__ import annotations

import argparse
from pathlib import Path


def parse_n_init(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--n_init must be >= 1 or 'auto'.")
    return parsed


def parse_ratio_schedule(value: str) -> list[tuple[int | None, float]]:
    value = value.strip()
    if not value:
        return []

    schedule: list[tuple[int | None, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "--adaptive_ratio_schedule entries must look like 'max_samples:ratio'."
            )
        max_samples_str, ratio_str = item.split(":", 1)
        max_samples_str = max_samples_str.strip().lower()
        ratio = float(ratio_str.strip())
        if ratio <= 0:
            raise argparse.ArgumentTypeError(
                "--adaptive_ratio_schedule ratios must be > 0."
            )
        if max_samples_str in {"inf", "none", "default"}:
            max_samples = None
        else:
            max_samples = int(max_samples_str)
            if max_samples < 1:
                raise argparse.ArgumentTypeError(
                    "--adaptive_ratio_schedule max_samples must be >= 1."
                )
        schedule.append((max_samples, ratio))

    finite_thresholds = [threshold for threshold, _ in schedule if threshold is not None]
    if finite_thresholds != sorted(finite_thresholds):
        raise argparse.ArgumentTypeError(
            "--adaptive_ratio_schedule thresholds must be sorted in ascending order."
        )
    if sum(1 for threshold, _ in schedule if threshold is None) > 1:
        raise argparse.ArgumentTypeError(
            "--adaptive_ratio_schedule can include at most one default 'inf' entry."
        )
    return schedule


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Sample representative feature indices with RAPIDS clustering. "
            "Surface trimmed to the live leaderboard recipes after cycles 1-31. "
            "See CONJECTURE.md for exhausted-family rationale."
        )
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Directory containing input .pt tensors with shape [N, D].")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory where output .npy index arrays will be written.")
    p.add_argument("--ratio", type=float, required=True,
                   help="Cluster ratio r. Number of clusters is int(r * N).")
    p.add_argument(
        "--adaptive_ratio_schedule",
        type=parse_ratio_schedule,
        default=[],
        help=(
            "Optional comma-separated per-slide ratio overrides in ascending "
            "'max_samples:ratio' order, e.g. '1000:0.025'. The first threshold "
            "with N <= max_samples wins; otherwise --ratio is used."
        ),
    )
    p.add_argument(
        "--min_clusters",
        type=int,
        default=1,
        help="Per-slide lower bound on cluster count (post-ratio, pre-budget cap).",
    )
    p.add_argument("--recursive", action="store_true",
                   help="Recursively scan input_dir for .pt files. Output is still flat.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .npy files. Default is to skip them.")
    p.add_argument("--device", type=str, default="cuda",
                   help="CUDA device for torch, e.g. cuda or cuda:0.")
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument(
        "--budget_ratio",
        type=float,
        default=None,
        help=(
            "Optional final sampling-budget ratio. When set, the saved index count is "
            "forced to int(budget_ratio * N), while --ratio continues to control the "
            "cluster count."
        ),
    )
    p.add_argument("--n_init", type=parse_n_init, default=1,
                   help="cuML KMeans n_init. Use an integer or 'auto'. Default: 1.")
    p.add_argument("--max_samples_per_batch", type=int, default=32768,
                   help="cuML KMeans max_samples_per_batch.")
    p.add_argument("--sample_block_size", type=int, default=65536,
                   help="GPU block size used to recover nearest sample indices.")
    p.add_argument(
        "--clustering",
        type=str,
        choices=["k_means", "k_medoid", "kcenter_greedy"],
        default="k_means",
        help=(
            "Clustering backend. 'k_means' uses cuML KMeans; 'k_medoid' uses GPU "
            "medoid refinement with actual sample centers; 'kcenter_greedy' "
            "bypasses clustering and selects target_selected patches via pure "
            "greedy farthest-first traversal (k-Center Greedy) on the input "
            "feature set — used as a coreset baseline."
        ),
    )
    p.add_argument(
        "--add_sample",
        type=str,
        choices=["none", "fps", "fps_seed_global", "fps_seed_global_cap", "facility_location", "random"],
        default="none",
        help=(
            "Per-cluster fill strategy. "
            "'fps' seeds farthest-point sampling with the cluster representative(s); "
            "'fps_seed_global' uses ALL globally-selected-so-far as the seed set "
            "(cross-cluster repulsion); "
            "'fps_seed_global_cap' keeps only the last --fps_seed_global_cap entries "
            "of the sorted (by patch-ID) global seed, i.e. the HIGH-patch-ID slice "
            "[-K:] — required for the 0.25 leaderboard recipe."
        ),
    )
    p.add_argument(
        "--fps_seed_global_cap",
        type=int,
        default=0,
        help=(
            "Maximum global-seed size per cluster iteration for "
            "--add_sample=fps_seed_global_cap (0 = no cap, degenerates to "
            "fps_seed_global)."
        ),
    )
    p.add_argument(
        "--add_sample_num",
        type=int,
        default=0,
        help="Per-non-empty-cluster fill count (only used when --budget_ratio is absent).",
    )
    p.add_argument(
        "--budget_allocation",
        type=str,
        choices=["uniform", "size", "sqrt", "entropy", "density", "adaptive_slide_size", "within_var", "within_var_sqrt"],
        default="uniform",
        help=(
            "How to distribute extra slots across clusters when --budget_ratio is set. "
            "'uniform' weights clusters equally; 'sqrt' weights by sqrt(available_count). "
            "'entropy' weights by per-cluster entropy -p*log(p) (p = cluster_size/N) with "
            "softmax temperature, redistributing budget toward mid-size clusters. "
            "All other allocation modes were closed in cycles 22-27 — see CONJECTURE.md."
        ),
    )
    p.add_argument(
        "--budget_allocation_power",
        type=float,
        default=0.5,
        help="Kept for signature compatibility. Not read under the three live allocation modes.",
    )
    p.add_argument(
        "--budget_allocation_entropy_temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature τ for --budget_allocation=entropy. "
            "weights = softmax(-p*log(p) / τ). Default 1.0."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_candidate_size",
        type=int,
        default=0,
        help=(
            "Bounded medoid-refinement shortlist size inside --clustering=k_medoid. "
            "0 disables and keeps the nearest-mean update."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_candidate_size_mode",
        choices=["fixed", "adaptive_sqrt"],
        default="fixed",
        help=(
            "'fixed': use --kmedoid_refine_candidate_size as a scalar for all clusters. "
            "'adaptive_sqrt': cs_k = clamp(round(sqrt(count_k)), floor, cap) per cluster."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_candidate_size_floor",
        type=int,
        default=4,
        help="Per-cluster minimum candidate_size in adaptive_sqrt mode.",
    )
    p.add_argument(
        "--kmedoid_refine_candidate_size_cap",
        type=int,
        default=16,
        help="Per-cluster maximum candidate_size in adaptive_sqrt mode.",
    )
    p.add_argument(
        "--kmedoid_refine_candidate_strategy",
        type=str,
        choices=["mean", "mean_margin", "knn_mean_margin", "margin_max", "local_density"],
        default="mean",
        help=(
            "Bounded k-medoid shortlist mode. 'mean' picks centroid-nearest; "
            "'mean_margin' prefers samples near own centroid but far from competing "
            "medoids (cycles 9-13 winner on 0.10 and 0.25); 'knn_mean_margin' "
            "pre-shortlists to 4*cs by mean_sq_dist then applies margin scoring "
            "on that subset only (faster O(N/4) cdist)."
        ),
    )
    p.add_argument(
        "--knn_mean_margin_pre_k_mult",
        type=float,
        default=4.0,
        help=(
            "Pre-shortlist multiplier for knn_mean_margin strategy. "
            "pre_k = min(max(int(round(candidate_size * mult)), candidate_size), count). "
            "Default 4.0 matches earlier defaults. Lower (2.0) = tighter shortlist; "
            "higher (8.0) = broader shortlist approaching full mean_margin."
        ),
    )
    p.add_argument(
        "--knn_mean_margin_score_mode",
        choices=["ratio", "subtractive"],
        default="ratio",
        help=(
            "knn_mean_margin score function. 'ratio' (default: "
            "margin_score = pre_mean_sq / nearest_competing_sq, pick smallest. "
            "'subtractive': margin_score = pre_mean_sq - lambda * nearest_competing_sq, "
            "pick smallest; additive-scale penalty rather than multiplicative."
        ),
    )
    p.add_argument(
        "--knn_mean_margin_subtractive_lambda",
        type=float,
        default=1.0,
        help="Lambda weight for subtractive score variant. Only active when score_mode=subtractive.",
    )
    p.add_argument(
        "--kmedoid_post_swap_polish_top_m",
        type=int,
        default=0,
        help=(
            "After k-medoid convergence, run ONE bounded global-swap polish pass on "
            "the top-M most contested medoids (smallest inter-medoid distance). "
            "0 = disabled (default). Applies to both budgets."
        ),
    )
    p.add_argument(
        "--kmedoid_post_swap_polish_n_passes",
        type=int,
        default=1,
        help=(
            "Number of polish passes to run after k-medoid convergence. "
            "Each pass recomputes inter-medoid distances on the current medoid set "
            "and re-runs bounded select on the top-M contested selections. Default 1 "
            ". 2-3 iterates the polish to convergence."
        ),
    )
    p.add_argument(
        "--kmedoid_post_swap_polish_criterion",
        choices=["closest_pair", "cluster_size", "centroid_dist", "silhouette", "within_variance"],
        default="closest_pair",
        help=(
            "Cluster-ranking rule for polish top-M selection. "
            "'closest_pair' (c24 default): rank by smallest min-inter-medoid-distance. "
            "'cluster_size': rank by largest cluster size (largest = more room for sub-optimal placement). "
            "'centroid_dist': rank by largest distance from medoid to cluster centroid."
        ),
    )
    p.add_argument(
        "--kmedoid_polish_pfseq_rerun_pct",
        type=float,
        default=0.0,
        help=(
            "After polish and after the first fvec+pfseq block, re-invoke one fvec+pfseq "
            "round with fresh delta-ranks on top-pct clusters. 0.0 = disabled. "
            "Requires kmedoid_post_swap_polish_top_m>0 and margin_swap_refine_passes>0."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_distance_metric",
        choices=["l2", "cosine"],
        default="l2",
        help=(
            "Distance metric used inside select_bounded_l2_medoid_index for candidate "
            "scoring. 'l2' (default): raw squared L2. 'cosine': features L2-normalized "
            "within this scoring step only; saved features and downstream eval unaffected."
        ),
    )
    p.add_argument(
        "--kmedoid_post_fvec_polish_top_m",
        type=int,
        default=0,
        help=(
            "After the fvec+pfseq margin_swap_refine block completes in pipeline.py, "
            "run one additional polish block on top-M most-contested medoids of the "
            "fvec-refined medoid set. DIFFERENT insertion timing from "
            "--kmedoid_post_swap_polish_top_m (which runs inside run_kmedoid, before fvec). "
            "0 = disabled."
        ),
    )
    p.add_argument(
        "--kmedoid_post_fvec_polish_relabel",
        type=int,
        default=0,
        help=(
            "c28 A v3: if 1, recompute cluster labels from post-fvec medoids before "
            "running the post-fvec polish block, preventing stale-label collisions. "
            "Also enables a defensive dedup pass after polish. 0 = disabled (."
        ),
    )
    p.add_argument(
        "--fps_extras_polish_enable",
        type=int,
        default=0,
        help=(
            "c28 B: if 1, run cluster-local FPS-extras polish after all medoid-side "
            "processing. Each extra is replaced with the within-cluster member that "
            "maximizes min-distance to the full current rep_idx set. 0 = disabled."
        ),
    )
    p.add_argument(
        "--kmedoid_multistart_criterion",
        type=str,
        choices=["inertia", "max_dispersion"],
        default="inertia",
        help=(
            "c28 C: multi-start selection criterion when kmedoid_n_init > 1. "
            "'inertia' (default): pick lowest-inertia run (prior behavior). "
            "'max_dispersion': pick run with largest min pairwise medoid distance "
            "(maximizes coverage diversity in distilled space)."
        ),
    )
    p.add_argument(
        "--fps_extras_reseed_after_refine",
        type=int,
        default=0,
        help=(
            "c28 D: if 1, discard pre-fvec FPS extras and re-run greedy-FPS using "
            "post-fvec medoids as seed. Requires --fps_extras_reseed_after_refine=1 "
            "and budget_ratio to be set (target_selected must be non-None). 0 = disabled."
        ),
    )
    p.add_argument(
        "--kmedoid_init_mode",
        choices=["random", "fps"],
        default="random",
        help=(
            "Initial medoid seeding. 'random' (default) = k-means warm-start. "
            "'fps' = farthest-point-sampling on full feature set to seed K medoids. "
            "Pre-rescope ABMIL closure may not apply to 9-MIL target."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_two_stage",
        type=int,
        default=0,
        help=(
            "If 1, first refine iteration uses 'mean_margin' (centroid-proximate seeds); "
            "subsequent iterations use the configured candidate_strategy. Default: 0 (single-strategy)."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_strategy_large_cluster",
        type=str,
        choices=["", "mean", "mean_margin", "margin_max", "knn_mean_margin", "local_density"],
        default="",
        help=(
            "Per-cluster strategy dispatch. If non-empty and cluster_size >= "
            "kmedoid_refine_strategy_size_adaptive_threshold, use this strategy instead of "
            "--kmedoid_refine_candidate_strategy for that cluster. Empty = disabled (default)."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_strategy_size_adaptive_threshold",
        type=int,
        default=0,
        help=(
            "Cluster-size threshold (inclusive) to switch to "
            "--kmedoid_refine_strategy_large_cluster. 0 = disabled (default). "
            "Typical values: 64, 128, 256."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_min_cluster_size",
        type=int,
        default=1,
        help=(
            "Per-cluster membership threshold. Clusters smaller than this keep the "
            "nearest-mean update even when --kmedoid_refine_candidate_size > 0."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_centroid_reg",
        type=float,
        default=0.0,
        help=(
            "Centroid regularization weight added to the bounded refine objective: "
            "within-cluster L2 cost + weight * cluster_size * centroid_distance_sq."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_separation_reg",
        type=float,
        default=0.0,
        help=(
            "Separation reward weight: objective -= weight * cluster_size * "
            "nearest_competing_center_distance_sq. 0.22 is the 0.10/0.25 leaderboard value."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_separation_reg_mode",
        choices=["fixed", "adaptive_sqrt"],
        default="fixed",
        help=(
            "'fixed': sep_reg is a scalar (default). "
            "'adaptive_sqrt': per-cluster sep_reg_k = sep_reg_base * "
            "clamp(sqrt(size_k/mean_size), scale_min, scale_max)."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_separation_reg_scale_min",
        type=float,
        default=0.5,
        help="Per-cluster adaptive sep_reg_k lower clamp multiplier. Default 0.5.",
    )
    p.add_argument(
        "--kmedoid_refine_separation_reg_scale_max",
        type=float,
        default=2.0,
        help="Per-cluster adaptive sep_reg_k upper clamp multiplier. Default 2.0.",
    )
    p.add_argument(
        "--margin_swap_refine_passes",
        type=int,
        default=0,
        help=(
            "Per-cluster margin-swap refinement passes. Objective (per pick x in "
            "cluster c): mean dist to other-cluster selected - mean dist to same-cluster "
            "unselected. Required for the 0.01 leaderboard recipe with sign=-1."
        ),
    )
    p.add_argument(
        "--margin_swap_adaptive_np_large_threshold",
        type=int,
        default=0,
        help=(
            "If > 0, clusters with size >= threshold receive (margin_swap_refine_passes + 1) "
            "swap passes; smaller clusters use the configured margin_swap_refine_passes. "
            "0 = disabled (uniform np). Rescues the 'np=3 global KILL' failure by "
            "restricting the extra pass to large clusters only."
        ),
    )
    p.add_argument(
        "--margin_swap_objective_sign",
        type=int,
        choices=[1, -1],
        default=1,
        help=(
            "Sign on margin-swap objective. +1 = forward (closed at all tested cells, "
            "+1 = standard direction; -1 = anti."
        ),
    )
    p.add_argument(
        "--margin_swap_min_cluster_size",
        type=int,
        default=2,
        help=(
            "Skip anti-ms swap on clusters with fewer than this many members. "
            "2=baseline (existing behavior). Higher values skip more small clusters, "
            "reducing K-loop overhead without affecting large-cluster swap quality. "
            "M=8 rescues cf=0.17 n=2 from DQ."
        ),
    )
    p.add_argument(
        "--margin_swap_cross_cluster_top_k",
        type=int,
        default=0,
        help=(
            "Augment each cluster's unselected swap-in pool with the top-K nearest "
            "unselected patches from OTHER clusters (measured by distance to this "
            "cluster's centroid). 0=intra-cluster-only (baseline). "
            "K=8 expands swap space without cf escalation."
        ),
    )
    p.add_argument(
        "--margin_swap_inter_cluster_spread_weight",
        type=float,
        default=0.0,
        help=(
            "If > 0, add a penalty to the anti-ms swap objective equal to "
            "-weight * min_dist_to_other_cluster_selections (reward spread). "
            "0.0 = disabled (default). Applies in the fvec code path."
        ),
    )
    p.add_argument(
        "--fps_metric",
        type=str,
        choices=["l2", "cosine"],
        default="l2",
        help=(
            "Distance metric for FPS-style add_sample fills. "
            "'l2' (default) uses squared L2; 'cosine' uses 1 - cos(x,y) on the "
            "sampling-side only — feature tensors are not modified downstream."
        ),
    )
    p.add_argument(
        "--fps_magnitude_beta",
        type=float,
        default=0.0,
        help=(
            "If > 0, weight per-pair FPS distance by (||x_i|| * ||x_j||)^(beta/2). "
            "Patches with larger feature magnitude get boosted distances. "
            "0.0 = no weighting (default; identity multiplier)."
        ),
    )
    p.add_argument(
        "--cdist_impl",
        type=str,
        choices=["torch", "matmul"],
        default="torch",
        help=(
            "Implementation for precomputed cdist in k-medoid refine. "
            "'matmul' uses the algebraic identity ||x-y||^2=||x||^2+||y||^2-2x@y.T "
            "for tensor-core acceleration. Default: 'torch'."
        ),
    )
    p.add_argument(
        "--cdist_fp16",
        type=int,
        default=0,
        help=(
            "If 1, cast inputs to fp16 before cdist in k-medoid refine to halve "
            "memory bandwidth. Output is cast back to fp32 after squaring. Default: 0."
        ),
    )
    p.add_argument(
        "--cdist_compile",
        type=int,
        default=0,
        help=(
            "If 1, wrap the N×K cdist precomputation in torch.compile(mode='reduce-overhead', "
            "dynamic=True) to reduce kernel-launch overhead. Falls back to uncompiled on error. "
            "Default: 0."
        ),
    )
    p.add_argument(
        "--refine_batched",
        type=int,
        default=0,
        help=(
            "If 1, use algebraic-identity batched medoid refine: eliminates per-cluster "
            "within-cluster cdist launches by exploiting sum_i||c_j-m_i||^2 = M*||c_j-mean||^2 + const. "
            "Produces identical medoid indices to the baseline path. Default: 0."
        ),
    )
    p.add_argument(
        "--fps_l2_normalize",
        type=int,
        default=0,
        help=(
            "If 1, L2-normalize features before FPS distance computation only (decoupled metric). "
            "Raw features are preserved downstream for k-medoid, refine, and eval. Default: 0."
        ),
    )
    p.add_argument(
        "--fps_hybrid_alpha",
        type=float,
        default=1.0,
        help=(
            "Blend L2_normalize and cosine FPS distances: d = α*d_L2_norm² + (1−α)*(1−cos). "
            "α=1.0 (default) = pure L2_normalize path (unchanged). Requires --fps_l2_normalize=1."
        ),
    )
    p.add_argument(
        "--margin_swap_vectorize_global_cdist",
        action="store_true",
        default=False,
        help=(
            "If set, precompute d(all_sel, all_sel) and d(samples, all_sel) globally "
            "before the per-cluster loop in swap_refine_margin_per_cluster, replacing "
            "per-cluster cdist launches with index_select lookups (Strategy A: recompute "
            "after each successful swap). Semantically equivalent; reduces K×n_passes "
            "small kernel launches. Default: off (legacy path).."
        ),
    )
    p.add_argument(
        "--margin_swap_top_m_unsel",
        type=int,
        default=0,
        help=(
            "If > 0, restrict the per-cluster unsel pool for anti-ms to the M nearest "
            "patches to the cluster centroid (L2 distance). Reduces cdist size from "
            "O(|unsel|) to O(M) per cluster. 0 = disabled (baseline).."
        ),
    )
    p.add_argument(
        "--margin_swap_vectorize_mode",
        type=str,
        choices=["A", "B"],
        default="A",
        help=(
            "Vectorize mode when --margin_swap_vectorize_global_cdist is set. "
            "'A' (default): rebuild global cdist after each swap (Strategy A, exact). "
            "'B': rebuild only at pass boundary (Strategy B, approximate, fewer rebuilds). "
            "."
        ),
    )
    p.add_argument(
        "--knn_swap_refine_k",
        type=int,
        default=0,
        help=(
            "If > 0, run a single-pass global k-NN swap refinement after "
            "swap_refine_margin_per_cluster. For each selected patch, tests swapping "
            "it with its k_nn nearest globally-unselected neighbors under a "
            "mean-dist-to-other-selected objective (anti-ms analog, cross-cluster). "
            "0 = disabled (default).."
        ),
    )
    p.add_argument(
        "--margin_swap_variance_skip_threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, skip clusters whose mean intra-distance is below "
            "threshold × global_median_intra_dist in swap_refine_margin_per_cluster. "
            "Tight clusters have low swap potential; skipping reduces Python K-loop "
            "iterations. 0.0 = disabled.."
        ),
    )
    p.add_argument(
        "--feature_random_projection_dim",
        type=int,
        default=0,
        help=(
            "If > 0, project feature tensor to this dimension via deterministic Gaussian "
            "random projection before clustering and all downstream distance computations. "
            "0 = disabled (default). Output indices are global patch IDs; eval side uses "
            "original CONCH features."
        ),
    )
    p.add_argument(
        "--margin_swap_adaptive_n",
        type=int,
        default=0,
        help=(
            "If 1, after pass 0 skip clusters that did not swap in the previous pass. "
            "Reduces K-loop iterations by 30-45%% on typical anti-ms convergence. "
            "Default 0 = disabled (all passes visit all clusters)."
        ),
    )
    p.add_argument(
        "--margin_swap_other_sel_approx",
        type=str,
        default="none",
        choices=["none", "centroid"],
        help=(
            "If 'centroid', replace other_sel patch pool (size O) with per-cluster "
            "centroids (size K-1) weighted by cluster size in the anti-ms margin. "
            "Default 'none'"
        ),
    )
    p.add_argument(
        "--margin_swap_batch_k",
        type=int,
        default=1,
        help=(
            "If >1, combine sel and unsel cdist calls into a single batched cdist "
            "(simplified batched-K: cat + split), saving 50%% of anti-ms cdist launches. "
            "Default 1 = legacy two-cdist path."
        ),
    )
    p.add_argument(
        "--margin_swap_chain_k",
        type=int,
        default=1,
        help=(
            "If >1, after applying the first per-cluster swap in anti-ms, search for "
            "a second swap on the updated selection (chain-k=2 move structure). "
            "Default 1 = single-swap."
        ),
    )
    p.add_argument(
        "--margin_swap_vectorize_full_kloop",
        type=int,
        default=0,
        help=(
            "If 1, replace per-cluster Python K-loop with fully vectorized scatter/masked-reduce "
            "ops over cluster labels. Attacks the 76%%-of-cost Python K-loop (c14 profile). "
            "Default 0 = per-cluster Python loop."
        ),
    )
    p.add_argument(
        "--margin_swap_jit_script",
        type=int,
        default=0,
        help=(
            "If 1, JIT-compile the cdist+mean ops via torch.jit.script for reduced "
            "Python-interpreter overhead. Default 0 = no JIT."
        ),
    )
    p.add_argument(
        "--margin_swap_agg_weight",
        type=str,
        default="uniform",
        choices=["uniform", "cluster_size", "uncertainty", "margin_magnitude"],
        help=(
            "Anti-ms other_sel aggregator weighting. 'uniform' = mean. "
            "'cluster_size' = 1/sqrt(size_of_cluster_containing(j)). "
            "'uncertainty' = 1/(d(j, centroid_of(cluster_of(j))) + eps). "
            "'margin_magnitude' = abs(margin(j)) weights discriminative patches higher. "
            "Default 'uniform' =."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_top_pct",
        type=float,
        default=0.0,
        help=(
            "If >0 AND margin_swap_vectorize_full_kloop=1, run sequential anti-ms on the "
            "top-pct clusters (by fvec pass-0 delta) after fvec passes complete. "
            "Recovers sequential-quality on worst-parallel-drift clusters. Default 0.0 = pure fvec."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_mode",
        type=str,
        default="single",
        choices=["single", "chain_k"],
        help=(
            "Sequential residual move structure: 'single' = single-swap per cluster, "
            "'chain_k' = chain-k=2 per cluster. Only active when post_fvec_seq_top_pct > 0. "
            "Default 'single'."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_select_mode",
        choices=["delta_rank", "inter_medoid_proximity"],
        default="delta_rank",
        help=(
            "pfseq cluster-selection criterion. 'delta_rank': "
            "select top-pct clusters by per-cluster fvec delta magnitude. "
            "'inter_medoid_proximity': select top-pct clusters with smallest "
            "inter-medoid minimum distance (most contested placement)."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_pass2_top_pct",
        type=float,
        default=0.0,
        help=(
            "If >0 AND post_fvec_seq_top_pct > 0 AND fvec active, run a SECOND pfseq pass "
            "on the next-ranked clusters (ranks top_pct to top_pct+pass2_top_pct by pass-0 delta). "
            "State-updated between passes (true iterative residual). "
            "0.0 (default) = single pfseq pass (c17 A behavior)."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_chain_k",
        type=int,
        default=2,
        help=(
            "Chain-k depth for the pfseq residual (only active when "
            "post_fvec_seq_mode=chain_k). 2 (default) =. "
            "3 = chain-k=3 (primary + 2 secondary swaps). Values outside {2,3} rejected."
        ),
    )
    p.add_argument(
        "--margin_swap_num_candidates",
        type=int,
        default=1,
        help=(
            "In sequential anti-ms, evaluate top-N sel-out and top-N unsel-in candidates "
            "per cluster (N^2 simplified-delta pairs) and apply best. Default 1 = current greedy."
        ),
    )
    p.add_argument(
        "--margin_swap_torchcompile_mode",
        type=str,
        default="none",
        choices=["none", "reduce-overhead"],
        help=(
            "If 'reduce-overhead', wrap fvec _run_one_fvec_step with "
            "torch.compile(mode='reduce-overhead'). "
            "Default 'none'."
        ),
    )
    p.add_argument(
        "--margin_swap_post_fvec_seq_pass2_mode",
        choices=["stale", "recomputed"],
        default="stale",
        help=(
            "Iterative pfseq pass-2 source. 'stale' uses pass-1 deltas at next-ranked "
            "clusters (c18 C behavior, CLOSED). 'recomputed' zeroes deltas and runs one "
            "more fvec step on current state, then takes fresh top-pct for pass 2. "
            "Only relevant when margin_swap_post_fvec_seq_pass2_top_pct > 0.0."
        ),
    )
    p.add_argument(
        "--margin_swap_fvec_microbatch_b",
        type=int,
        default=0,
        help=(
            "If >=2 AND margin_swap_vectorize_full_kloop=1, split each fvec pass into B "
            "random-permuted cluster batches with selected_mask state updates BETWEEN "
            "batches. Emulates sequential-ordering within each pass while retaining "
            "vectorization inside each batch. 0 (default) = pure fvec (all K at once). "
            "B=num_clusters approaches pure sequential."
        ),
    )
    p.add_argument(
        "--margin_swap_chain_k_top_pct",
        type=float,
        default=0.0,
        help=(
            "If >0 AND margin_swap_chain_k>=2, apply chain-k-length moves only to top-pct "
            "of clusters ranked by pass-1 best-swap margin-delta. Saves (1-pct) of chain-k "
            "extra cost. Default 0.0 = chain-k applied to all clusters (c14 F behavior)."
        ),
    )
    p.add_argument(
        "--post_antims_max_spread_k",
        type=int,
        default=0,
        help=(
            "If >0, run this many additional per-cluster greedy max-spread swap passes "
            "AFTER anti-ms converges. Maximizes mean distance to current selected pool. "
            "Default 0 = disabled."
        ),
    )
    p.add_argument(
        "--profile_anti_ms",
        type=int,
        default=0,
        help=(
            "If 1, emit [ANTI_MS_PROFILE] per-op GPU wall-clock breakdown for anti-ms "
            "K-loop (cdist / agg / swap_search). Measurement-only. Default 0 = disabled."
        ),
    )
    p.add_argument(
        "--kmedoid_n_init",
        type=int,
        default=1,
        help=(
            "If >1, run k-medoid init this many times with different random seeds and "
            "keep the lowest-inertia result. Default 1 = single init."
        ),
    )
    p.add_argument(
        "--margin_swap_other_sel_subsample",
        type=int,
        default=0,
        help=(
            "If > 0, randomly subsample the other-cluster selected pool to this size "
            "per anti-ms pass before computing d_sel_other / d_unsel_other. "
            "Same subsample used for all clusters in one pass (per-pass generator). "
            "Default 0 = disabled (use full other-cluster selected pool)."
        ),
    )
    p.add_argument(
        "--margin_swap_margin_agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help=(
            "Aggregator for margin computation over the other-cluster selected set. "
            "'mean'; 'median' (robust to outlier distances). "
            "median approximates the algebraic update formula."
        ),
    )
    p.add_argument(
        "--feature_pca_projection_dim",
        type=int,
        default=0,
        help=(
            "If > 0 and < D, project feature tensor to this dimension via cached PCA "
            "(variance-preserving basis; bootstrapped from the first slide). "
            "Placed inside the T_v fence so PCA cost is included in sampler time. "
            "Cannot be combined with --feature_random_projection_dim. "
            "Default 0 = disabled."
        ),
    )
    p.add_argument(
        "--time_only",
        action="store_true",
        default=False,
        help=(
            "Emit [gpu-time] markers per slide and exit without writing index files "
            "or invoking downstream eval. Used for GPU-only T_k ruler measurement."
        ),
    )
    p.add_argument(
        "--sort_indices",
        action="store_true",
        default=True,
        help="Sort sampled indices into original sample order before saving.",
    )
    p.add_argument(
        "--no_sort_indices",
        dest="sort_indices",
        action="store_false",
        help="Keep centroid order instead of original sample order.",
    )
    p.add_argument("--mmap", action="store_true", default=True,
                   help="Use torch.load(..., mmap=True). Enabled by default.")
    p.add_argument("--no_mmap", dest="mmap", action="store_false")
    # --- Cycle-29: four new mechanism axes ------------------------------------
    p.add_argument(
        "--kmedoid_refine_margin_dir_1d",
        type=int,
        default=0,
        help=(
            "c29 A: if 1, replace full-D margin score with a 1D projection onto the "
            "direction from cluster centroid to its nearest-other-centroid. Applies to "
            "margin_max and knn_mean_margin candidate strategies."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_sep_reg_density",
        type=int,
        default=0,
        help=(
            "c29 B: if 1, compute per-cluster density-adaptive sep_reg in "
            "[sep_reg_low, sep_reg_high]. Dense clusters get higher sep_reg; "
            "sparse clusters get lower sep_reg."
        ),
    )
    p.add_argument(
        "--kmedoid_refine_sep_reg_low",
        type=float,
        default=0.15,
        help="c29 B: lower bound for density-adaptive sep_reg. Default 0.15.",
    )
    p.add_argument(
        "--kmedoid_refine_sep_reg_high",
        type=float,
        default=0.28,
        help="c29 B: upper bound for density-adaptive sep_reg. Default 0.28.",
    )
    p.add_argument(
        "--kmedoid_post_swap_polish_forward_select_k",
        type=int,
        default=1,
        help=(
            "c29 C: forward-selection candidate count for polish step. "
            "1=argmax via select_bounded_l2_medoid_index (default, current behavior); "
            ">1=take top-K nearest-to-mean candidates then pick the one that maximizes "
            "min pairwise distance to other medoids (forward-select)."
        ),
    )
    p.add_argument(
        "--fps_aggregate_mode",
        type=str,
        choices=["min", "median"],
        default="min",
        help=(
            "c29 D: FPS greedy aggregate function. 'min'=max-min-distance (default). "
            "'median'=max-median-distance over the initial seed set (more robust to outlier seeds)."
        ),
    )
    return p.parse_args()


def validate_args(args) -> None:
    if args.ratio <= 0:
        raise ValueError(f"--ratio must be > 0, got {args.ratio}")
    if args.min_clusters < 1:
        raise ValueError("--min_clusters must be >= 1.")
    if args.budget_ratio is not None and args.budget_ratio <= 0:
        raise ValueError(f"--budget_ratio must be > 0, got {args.budget_ratio}")
    if args.sample_block_size < 1:
        raise ValueError("--sample_block_size must be >= 1.")
    if args.kmedoid_refine_candidate_size < 0:
        raise ValueError("--kmedoid_refine_candidate_size must be >= 0.")
    if args.kmedoid_refine_min_cluster_size < 1:
        raise ValueError("--kmedoid_refine_min_cluster_size must be >= 1.")
    if args.kmedoid_refine_centroid_reg < 0:
        raise ValueError("--kmedoid_refine_centroid_reg must be >= 0.")
    if args.kmedoid_refine_separation_reg < 0:
        raise ValueError("--kmedoid_refine_separation_reg must be >= 0.")
    if args.margin_swap_refine_passes < 0:
        raise ValueError("--margin_swap_refine_passes must be >= 0.")
    if args.margin_swap_min_cluster_size < 1:
        raise ValueError("--margin_swap_min_cluster_size must be >= 1.")
    if args.margin_swap_cross_cluster_top_k < 0:
        raise ValueError("--margin_swap_cross_cluster_top_k must be >= 0.")
    if args.max_iter < 1:
        raise ValueError("--max_iter must be >= 1.")
    if args.max_samples_per_batch < 1:
        raise ValueError("--max_samples_per_batch must be >= 1.")
    if args.add_sample_num < 0:
        raise ValueError("--add_sample_num must be >= 0.")
    if (
        args.budget_ratio is not None
        and args.add_sample == "none"
        and not args.time_only
        and args.clustering != "kcenter_greedy"
    ):
        raise ValueError(
            "--budget_ratio requires --add_sample in "
            "{fps, fps_seed_global, fps_seed_global_cap, facility_location}."
        )
    if args.add_sample == "fps_seed_global_cap" and args.fps_seed_global_cap <= 0:
        raise ValueError(
            "--add_sample=fps_seed_global_cap requires --fps_seed_global_cap > 0."
        )
    if args.feature_random_projection_dim < 0:
        raise ValueError("--feature_random_projection_dim must be >= 0.")
    if getattr(args, "margin_swap_adaptive_n", 0) not in {0, 1}:
        raise ValueError("--margin_swap_adaptive_n must be 0 or 1.")
    if getattr(args, "margin_swap_other_sel_subsample", 0) < 0:
        raise ValueError("--margin_swap_other_sel_subsample must be >= 0.")
    if getattr(args, "feature_pca_projection_dim", 0) < 0:
        raise ValueError("--feature_pca_projection_dim must be >= 0.")
    if getattr(args, "margin_swap_batch_k", 1) < 1:
        raise ValueError("--margin_swap_batch_k must be >= 1.")
    if getattr(args, "margin_swap_chain_k", 1) not in {1, 2}:
        raise ValueError("--margin_swap_chain_k must be 1 or 2.")
    if getattr(args, "post_antims_max_spread_k", 0) < 0:
        raise ValueError("--post_antims_max_spread_k must be >= 0.")
    if getattr(args, "profile_anti_ms", 0) not in {0, 1}:
        raise ValueError("--profile_anti_ms must be 0 or 1.")
    if getattr(args, "kmedoid_n_init", 1) < 1:
        raise ValueError("--kmedoid_n_init must be >= 1.")
    if args.feature_random_projection_dim > 0 and getattr(args, "feature_pca_projection_dim", 0) > 0:
        raise ValueError(
            "--feature_random_projection_dim and --feature_pca_projection_dim cannot both be non-zero."
        )
    if getattr(args, "margin_swap_vectorize_full_kloop", 0) not in {0, 1}:
        raise ValueError("--margin_swap_vectorize_full_kloop must be 0 or 1.")
    if getattr(args, "margin_swap_fvec_microbatch_b", 0) < 0:
        raise ValueError("--margin_swap_fvec_microbatch_b must be >= 0.")
    if getattr(args, "margin_swap_jit_script", 0) not in {0, 1}:
        raise ValueError("--margin_swap_jit_script must be 0 or 1.")
    if getattr(args, "margin_swap_agg_weight", "uniform") not in {"uniform", "cluster_size", "uncertainty", "margin_magnitude"}:
        raise ValueError("--margin_swap_agg_weight must be uniform, cluster_size, uncertainty, or margin_magnitude.")
    _top_pct = getattr(args, "margin_swap_chain_k_top_pct", 0.0)
    if not (0.0 <= _top_pct <= 1.0):
        raise ValueError("--margin_swap_chain_k_top_pct must be in [0.0, 1.0].")
    _pfseq_top_pct = getattr(args, "margin_swap_post_fvec_seq_top_pct", 0.0)
    if not (0.0 <= _pfseq_top_pct <= 1.0):
        raise ValueError("--margin_swap_post_fvec_seq_top_pct must be in [0.0, 1.0].")
    if getattr(args, "margin_swap_post_fvec_seq_mode", "single") not in {"single", "chain_k"}:
        raise ValueError("--margin_swap_post_fvec_seq_mode must be single or chain_k.")
    if getattr(args, "margin_swap_num_candidates", 1) < 1:
        raise ValueError("--margin_swap_num_candidates must be >= 1.")
    if getattr(args, "margin_swap_torchcompile_mode", "none") not in {"none", "reduce-overhead"}:
        raise ValueError("--margin_swap_torchcompile_mode must be none or reduce-overhead.")
