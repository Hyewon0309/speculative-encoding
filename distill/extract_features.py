#!/usr/bin/env python
"""Run a patch encoder over WSI patches and write per-slide
``[N_patches, D]`` ``.pt`` feature tensors in the CLAM-0402 layout.

Either backend works:

  --checkpoint_path  <student.pt>      ← the distilled student you trained
  --teacher_model    <conchv15|uni|...> ← the original teacher (HF download)

Pick one. ``--teacher_model_path`` and ``--conchv15_use_projection`` /
``--no_conchv15_use_projection`` are forwarded to ``distill_lib.teacher.load_teacher``
when ``--teacher_model`` is set.

The script consumes pre-computed coord ``.npy`` files (one per slide,
structured array with fields ``x``, ``y``, ``tile_size_lv0``) — i.e. the
output of ``CLAM`` (``create_patches_fp.py``) or ``TRIDENT``. Patches are
read on the fly from the raw WSI via ``openslide``, normalised, batched,
and forwarded through the encoder. Output files mirror the directory tree
of the coord root so they slot into ``$FEATURE_ROOT`` (teacher) or
``$DISTILLED_FEATURE_ROOT`` (student) for the sampler.

Usage
-----

    # Distilled student
    python distill/extract_features.py \\
        --checkpoint_path  outputs/distilled_models/<run>/checkpoint_*.pt \\
        --wsi_root         /data/raw_wsi/cm16 \\
        --coord_root       /data/CLAM_0402/patch_coords/ps512/cm16 \\
        --output_root      /data/distilled/patch_features/ps512/distilled_cls/cm16

    # Original teacher (CONCH v1.5)
    python distill/extract_features.py \\
        --teacher_model    conchv15 \\
        --wsi_root         /data/raw_wsi/cm16 \\
        --coord_root       /data/CLAM_0402/patch_coords/ps512/cm16 \\
        --output_root      /data/CLAM_0402/patch_features/ps512/conch_v1_5/cm16

Multi-GPU: launch with ``torchrun --nproc_per_node=N`` and the slide list
is sharded round-robin across ranks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from distill_lib.student import load_student_from_checkpoint
from distill_lib.teacher import load_teacher


WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_path", type=Path, default=None,
                     help="Distilled student checkpoint (output of scripts/distill.sh).")
    src.add_argument("--teacher_model", type=str, default=None,
                     choices=["uni", "conch", "conchv15", "conch_v15", "conchv1_5",
                              "virchow", "prism", "provgigapath", "prov_gigapath",
                              "prov-gigapath", "gigapath",
                              "biomedclip", "openaiclip", "plip", "medsiglip"],
                     help="Run the original teacher encoder instead of a distilled student. "
                          "Mutually exclusive with --checkpoint_path.")
    p.add_argument("--teacher_model_path", type=str, default=None,
                   help="Optional local model dir / HF id passed to load_teacher.")
    p.add_argument("--conchv15_use_projection", dest="conchv15_use_projection",
                   action="store_true", default=True,
                   help="For --teacher_model conchv15: include the contrastive projection (default: on).")
    p.add_argument("--no_conchv15_use_projection", dest="conchv15_use_projection",
                   action="store_false")
    p.add_argument("--wsi_root", type=Path, required=True,
                   help="Directory containing the raw WSI files (recursive).")
    p.add_argument("--coord_root", type=Path, required=True,
                   help="Directory of coord .npy files (CLAM-0402 layout). "
                        "Slide ids = filename stems must match the WSI files.")
    p.add_argument("--output_root", type=Path, required=True,
                   help="Where to write per-slide .pt features. Layout mirrors "
                        "--coord_root.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Patches per forward pass.")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers per slide.")
    p.add_argument("--precision", type=str, default="fp16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cuda:0 / cpu. Default: auto.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .pt files.")
    p.add_argument("--log_every", type=int, default=50,
                   help="Log every N patches per slide.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers (torchrun)
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> Tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Single-process → (0, 1, 0)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


# ─────────────────────────────────────────────────────────────────────────────
# Data: WSI patch dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_coords(coord_path: Path) -> Tuple[np.ndarray, int]:
    """Load CLAM coord .npy. Returns (coords[N,2] (x,y, level-0 px), tile_size_lv0)."""
    arr = np.load(coord_path, allow_pickle=True)
    coords = np.stack([arr["x"], arr["y"]], axis=1).astype(np.int64)
    tile_size_lv0 = int(arr["tile_size_lv0"][0])
    return coords, tile_size_lv0


def find_wsi(wsi_root: Path, slide_id: str) -> Optional[Path]:
    """Resolve slide_id → raw WSI file under wsi_root (recursive search)."""
    for ext in WSI_EXTENSIONS:
        cand = wsi_root / f"{slide_id}{ext}"
        if cand.exists():
            return cand
    matches = [p for p in wsi_root.rglob(f"{slide_id}.*") if p.suffix.lower() in WSI_EXTENSIONS]
    return matches[0] if matches else None


class WSIPatchDataset(torch.utils.data.Dataset):
    """Lazy patch reader: opens the WSI in __getitem__ workers, returns a normalised tensor."""

    def __init__(self, wsi_path: Path, coords: np.ndarray, tile_size_lv0: int,
                 student_input_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.wsi_path = wsi_path
        self.coords = coords
        self.tile_size_lv0 = tile_size_lv0
        self.student_input_size = student_input_size
        # Per-worker WSI handle (openslide is not picklable across processes).
        self._slide = None
        # ToTensor + Normalize. Resize (if needed) is done per patch.
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def _ensure_slide(self):
        if self._slide is None:
            import openslide
            self._slide = openslide.OpenSlide(str(self.wsi_path))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_slide()
        x, y = self.coords[idx]
        patch = self._slide.read_region(
            (int(x), int(y)), 0,
            (self.tile_size_lv0, self.tile_size_lv0),
        ).convert("RGB")
        if self.tile_size_lv0 != self.student_input_size:
            patch = patch.resize(
                (self.student_input_size, self.student_input_size),
                Image.LANCZOS,
            )
        # uint8 → float [0, 1] → normalise
        arr = torch.from_numpy(np.asarray(patch, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return self.normalize(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_slide(student: torch.nn.Module, ds: WSIPatchDataset,
                 batch_size: int, num_workers: int,
                 device: torch.device, autocast_ctx) -> torch.Tensor:
    """Forward all patches of one slide and return ``[N, D]`` CLS features on CPU."""
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    feats: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast_ctx:
            out = student(batch)
        # ImageEncoderWrapper returns the final hidden state; the CLS token is at index 0.
        if isinstance(out, dict):
            out = out.get("cls", out.get("features"))
        if out.dim() == 3:               # [B, 1+N, D] → CLS
            out = out[:, 0, :]
        feats.append(out.float().cpu())
    return torch.cat(feats, dim=0)


def get_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        from contextlib import nullcontext
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_encoder(args, device: torch.device):
    """Return (encoder, input_size_px, mean, std).

    Distilled student   → loaded via ``load_student_from_checkpoint``.
    Original teacher    → loaded via ``distill_lib.teacher.load_teacher``.
    """
    if args.checkpoint_path is not None:
        enc = load_student_from_checkpoint(str(args.checkpoint_path)).to(device).eval()
        size = getattr(enc, "img_size", None) or (
            getattr(getattr(enc, "patch_embed", None), "img_size", (224, 224))[0]
        )
        return enc, int(size), IMAGENET_MEAN, IMAGENET_STD

    # Teacher path. distill_lib.teacher.load_teacher takes an args namespace and
    # reads .teacher_model / .teacher_model_path / .conchv15_use_projection.
    teacher_args = argparse.Namespace(
        teacher_model=args.teacher_model,
        teacher_model_path=args.teacher_model_path,
        teacher_model_name=None,
        conchv15_use_projection=args.conchv15_use_projection,
    )
    teacher, _eval_pre, _distill_pre, meta = load_teacher(teacher_args)
    teacher = teacher.to(device).eval()
    size = int(meta.get("img_size", 224))
    mean = tuple(meta.get("mean", IMAGENET_MEAN))
    std = tuple(meta.get("std", IMAGENET_STD))
    return teacher, size, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    device = torch.device(args.device) if args.device else (
        torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    )

    if is_main:
        src = args.checkpoint_path or f"teacher:{args.teacher_model}"
        print(f"[extract] encoder      = {src}")
        print(f"[extract] wsi_root     = {args.wsi_root}")
        print(f"[extract] coord_root   = {args.coord_root}")
        print(f"[extract] output_root  = {args.output_root}")
        print(f"[extract] world_size   = {world_size}, device = {device}")

    # ── Load encoder (student checkpoint OR original teacher) ───────────────
    encoder, encoder_input_size, encoder_mean, encoder_std = _load_encoder(args, device)
    if is_main:
        print(f"[extract] encoder input size = {encoder_input_size}px  "
              f"mean={encoder_mean}  std={encoder_std}")

    autocast_ctx = get_autocast(device, args.precision)

    # ── Enumerate slides ────────────────────────────────────────────────────
    coord_files = sorted(args.coord_root.rglob("*.npy"))
    if not coord_files:
        raise SystemExit(f"No .npy coord files under {args.coord_root}")
    if is_main:
        print(f"[extract] {len(coord_files)} coord files; sharding across {world_size} rank(s)")

    args.output_root.mkdir(parents=True, exist_ok=True)

    # ── Per-slide loop ──────────────────────────────────────────────────────
    n_done, n_skipped, n_missing = 0, 0, 0
    t0 = time.time()
    for idx, coord_path in enumerate(coord_files):
        if idx % world_size != rank:
            continue

        slide_id = coord_path.stem
        rel = coord_path.relative_to(args.coord_root)
        out_path = (args.output_root / rel).with_suffix(".pt")

        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        wsi_path = find_wsi(args.wsi_root, slide_id)
        if wsi_path is None:
            print(f"[rank {rank}] [warn] WSI not found for slide_id={slide_id}; skipping")
            n_missing += 1
            continue

        coords, tile = load_coords(coord_path)
        ds = WSIPatchDataset(
            wsi_path=wsi_path,
            coords=coords,
            tile_size_lv0=tile,
            student_input_size=encoder_input_size,
            mean=encoder_mean,
            std=encoder_std,
        )
        try:
            feats = encode_slide(
                student=encoder, ds=ds,
                batch_size=args.batch_size, num_workers=args.num_workers,
                device=device, autocast_ctx=autocast_ctx,
            )
        except Exception as exc:
            print(f"[rank {rank}] [error] slide {slide_id}: {type(exc).__name__}: {exc}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(feats, tmp)
        os.replace(tmp, out_path)
        n_done += 1

        if n_done % max(1, args.log_every // 10) == 0 or idx == len(coord_files) - 1:
            elapsed = time.time() - t0
            print(f"[rank {rank}] {n_done} written, {n_skipped} skipped, "
                  f"{n_missing} missing | {elapsed:.0f}s")

    print(f"[rank {rank}] done. wrote={n_done}  skipped={n_skipped}  missing={n_missing}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
