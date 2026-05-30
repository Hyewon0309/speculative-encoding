from __future__ import annotations

from pathlib import Path

import torch


def list_pt_files(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pattern = "**/*.pt" if recursive else "*.pt"
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")
    return files


def validate_flat_output_names(files: list[Path]) -> None:
    seen = {}
    for path in files:
        output_name = f"{path.stem}.npy"
        previous = seen.get(output_name)
        if previous is not None:
            raise ValueError(
                "Output directory is flat, but multiple input files would map to the same output "
                f"name {output_name}: {previous} and {path}"
            )
        seen[output_name] = path


def load_feature_tensor(path: Path, use_mmap: bool) -> torch.Tensor:
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if use_mmap:
        load_kwargs["mmap"] = True

    try:
        payload = torch.load(path, **load_kwargs)
    except Exception:
        if not use_mmap:
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(payload)}")
    if payload.ndim != 2:
        raise ValueError(f"Expected a 2D tensor in {path}, got shape {tuple(payload.shape)}")
    if payload.is_sparse:
        payload = payload.to_dense()

    return payload.detach().to(dtype=torch.float32).contiguous()
