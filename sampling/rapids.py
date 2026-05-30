from __future__ import annotations

import torch


def import_rapids():
    try:
        import cupy as cp
        from cuml.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "RAPIDS imports failed. Install via "
            "`pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 cupy-cuda12x`."
        ) from exc
    return cp, KMeans


def torch_to_cupy(tensor: torch.Tensor, cp):
    if not tensor.is_cuda:
        raise ValueError("torch_to_cupy expects a CUDA tensor.")
    try:
        return cp.from_dlpack(tensor)
    except Exception:
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def ensure_cupy_array(array, cp):
    if isinstance(array, cp.ndarray):
        return array
    if hasattr(array, "to_output"):
        return array.to_output("cupy")
    return cp.asarray(array)


def cupy_to_torch(array) -> torch.Tensor:
    try:
        return torch.utils.dlpack.from_dlpack(array)
    except Exception:
        return torch.utils.dlpack.from_dlpack(array.toDlpack())


def build_kmeans(num_clusters: int, args, KMeans, max_iter: int | None = None):
    return KMeans(
        n_clusters=num_clusters,
        max_iter=args.max_iter if max_iter is None else max_iter,
        tol=args.tol,
        random_state=args.random_state,
        init="scalable-k-means++",
        n_init=args.n_init,
        max_samples_per_batch=args.max_samples_per_batch,
        output_type="cupy",
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
