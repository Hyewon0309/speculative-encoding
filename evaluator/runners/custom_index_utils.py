from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def infer_custom_index_path(
    feature_path: Path,
    feature_root: Path,
    custom_index_root: Path,
) -> Optional[Path]:
    try:
        rel_path = feature_path.relative_to(feature_root)
        candidate = (custom_index_root / rel_path).with_suffix(".npy")
        if candidate.exists():
            return candidate
    except ValueError:
        pass

    flat_candidate = custom_index_root / f"{feature_path.stem}.npy"
    if flat_candidate.exists():
        return flat_candidate

    matches = sorted(custom_index_root.rglob(f"{feature_path.stem}.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple custom index files matched slide {feature_path.stem}: {matches}"
        )
    return None


def load_custom_indices(index_path: Path, n_features: Optional[int] = None) -> np.ndarray:
    arr = np.load(index_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, dict):
        if "indices" in arr:
            arr = arr["indices"]
        else:
            raise ValueError(f"Unsupported custom index dict keys in {index_path}: {arr.keys()}")
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Custom index file is empty: {index_path}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Custom indices must be integer dtype: {index_path}")
    arr = arr.astype(np.int64, copy=False)
    if n_features is not None:
        arr = arr[(arr >= 0) & (arr < n_features)]
    if arr.size == 0:
        raise ValueError(f"No valid custom indices remain after bounds check: {index_path}")
    return arr


def build_custom_index_cache(
    paths,
    feature_root: Path,
    custom_index_root: Optional[Path],
) -> Dict[str, np.ndarray]:
    if custom_index_root is None:
        return {}

    cache: Dict[str, np.ndarray] = {}
    flat_paths: List[Path] = []
    for path in paths:
        if isinstance(path, list):
            flat_paths.extend(path)
        else:
            flat_paths.append(path)

    for path in flat_paths:
        slide_id = path.stem
        if slide_id in cache:
            continue
        index_path = infer_custom_index_path(path, feature_root=feature_root, custom_index_root=custom_index_root)
        if index_path is None:
            continue
        cache[slide_id] = load_custom_indices(index_path)
    return cache


def get_custom_indices(
    custom_index_cache: Optional[Dict[str, np.ndarray]],
    slide_id: str,
    n_features: int,
) -> Optional[np.ndarray]:
    if not custom_index_cache or slide_id not in custom_index_cache:
        return None
    arr = np.asarray(custom_index_cache[slide_id]).reshape(-1).astype(np.int64, copy=False)
    arr = arr[(arr >= 0) & (arr < n_features)]
    if arr.size == 0:
        raise ValueError(f"Custom indices for slide {slide_id} are empty after bounds check")
    return arr
