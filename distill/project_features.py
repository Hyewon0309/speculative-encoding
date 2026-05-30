"""Project a tree of ``.pt`` feature tensors through a trained MLP projector.

Input layout:
    <input_dir>/.../<name>.pt   # torch.Tensor of shape (N, in_dim)

Output layout (mirrored):
    <output_dir>/.../<name>.pt  # torch.Tensor of shape (N, out_dim)

If the checkpoint was produced by ``train_mlp_projector.py``, its
architecture (in_dim, out_dim, hidden_dim, activation) is loaded
automatically — no flags needed.

Multi-GPU: launch with ``torchrun --nproc_per_node=N`` and the file
list is sharded across ranks. Single-process launch also works.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from distill_lib.mlp_projector import load_mlp_projector


def _parse_args():
    p = argparse.ArgumentParser(description="Project features through a trained MLP projector.")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Source directory. Scans recursively for .pt files.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Destination directory. Structure mirrors --input_dir.")
    p.add_argument("--projector_path", type=str, required=True,
                   help="Path to an MLP checkpoint written by train_mlp_projector.py.")
    p.add_argument("--chunk_size", type=int, default=4096,
                   help="Rows per forward pass. Set smaller if OOM.")
    p.add_argument("--precision", type=str, default="fp32",
                   choices=["fp32", "fp16", "bf16"],
                   help="Autocast dtype. Output is always cast back to the input dtype.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-project files whose output already exists.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device. Default: cuda if available, else cpu.")
    p.add_argument("--log_every", type=int, default=50,
                   help="Log progress every N files.")
    return p.parse_args()


def _init_distributed():
    """Return (rank, world_size) for torchrun launches, else (0, 1)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _scan_pt_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pt") if p.is_file())


def _autocast_ctx(device: torch.device, precision: str):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


@torch.no_grad()
def _project_file(
    src: Path,
    dst: Path,
    projector: torch.nn.Module,
    device: torch.device,
    chunk_size: int,
    precision: str,
) -> tuple[int, int]:
    """Load ``src``, project chunk-by-chunk, write ``dst``. Returns (N, out_dim)."""
    feats = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(feats, torch.Tensor):
        raise TypeError(
            f"{src}: expected a torch.Tensor, got {type(feats).__name__}"
        )
    if feats.dim() != 2:
        raise ValueError(
            f"{src}: expected 2-D tensor (N, D), got shape {tuple(feats.shape)}"
        )

    orig_dtype = feats.dtype
    n = feats.size(0)
    if n == 0:
        out_dim = int(projector.out_dim)
        projected = torch.empty(0, out_dim, dtype=orig_dtype)
    else:
        chunks: list[torch.Tensor] = []
        autocast = _autocast_ctx(device, precision)
        for i in range(0, n, chunk_size):
            x = feats[i : i + chunk_size].to(device, non_blocking=True).float()
            with autocast:
                y = projector(x)
            chunks.append(y.float().cpu())
        projected = torch.cat(chunks, dim=0).to(orig_dtype)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(projected, tmp)
    os.replace(tmp, dst)
    return n, int(projected.size(1))


def main():
    args = _parse_args()
    rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input_dir does not exist: {input_dir}")
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    projector = load_mlp_projector(args.projector_path, map_location=device).to(device)
    projector.eval()
    in_dim = projector.in_dim
    out_dim = projector.out_dim
    if is_main:
        print(
            f"[rank 0] projector: {in_dim} → {out_dim} "
            f"(hidden={projector.hidden_dim}, layers={projector.num_hidden_layers}, "
            f"act={projector.activation})",
            flush=True,
        )

    all_files = _scan_pt_files(input_dir)
    if is_main:
        print(f"[rank 0] found {len(all_files)} .pt files under {input_dir}", flush=True)

    # Shard by rank (round-robin keeps per-rank file sizes balanced on average).
    my_files = all_files[rank::world_size]

    start = time.time()
    total_rows = 0
    skipped = 0
    for idx, src in enumerate(my_files, 1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            n, d = _project_file(src, dst, projector, device, args.chunk_size, args.precision)
        except Exception as exc:
            raise RuntimeError(f"Failed on {src}: {exc}") from exc
        total_rows += n
        if d != out_dim:
            raise RuntimeError(
                f"{src}: projector produced out_dim={d}, expected {out_dim}"
            )
        if idx % args.log_every == 0 or idx == len(my_files):
            elapsed = time.time() - start
            rate = total_rows / max(elapsed, 1e-6)
            print(
                f"[rank {rank}] {idx}/{len(my_files)} | rows={total_rows} | "
                f"skipped={skipped} | {rate:,.0f} rows/s | {elapsed:.1f}s",
                flush=True,
            )

    if is_main:
        print(f"[rank 0] done. world_size={world_size}", flush=True)


if __name__ == "__main__":
    main()
