"""Self-contained TITAN slide-encoder wrapper.

TITAN (Mahmood Lab, ICLR 2025) consumes pre-extracted CONCH v1.5 patch
features and produces a single slide-level embedding. The official
HuggingFace checkpoint ships its own modeling code, so the only thing
this wrapper does is:

    1. ``AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)``
    2. ``model.encode_slide_from_patch_features(features, coords, patch_size_lv0)``

This file is intentionally kept dependency-free (only ``torch`` and
``transformers``) so the public release does not pull in the internal
``hyperpvlm`` package.

Inputs
------
``features`` : (N, C) or (B, N, C) — CONCH v1.5 patch embeddings (C=768).
``coords``   : (N, 2) or (B, N, 2) — top-left pixel coords at level-0.
``patch_size_lv0`` : int — patch side length in level-0 pixels (e.g. 512).

Output
------
``slide_embedding`` : (B, D)  — TITAN slide embedding (D = 768 by default).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TITAN(nn.Module):
    """TITAN slide encoder wrapper.

    Parameters
    ----------
    model_dir
        Optional local directory holding a TITAN checkpoint. When ``None``
        (the default), the model is fetched from HuggingFace using the
        repository identified by the env var ``TITAN_HF_REPO`` (falls back
        to ``MahmoodLab/TITAN``).
    device
        Torch device for inference (``"cuda"`` / ``"cuda:0"`` / ``"cpu"``).
    no_proj
        Forwarded to ``encode_slide_from_patch_features`` when the model
        version supports it. Keep ``True`` to skip the contrastive
        projection (which is fitted for vision-language alignment, not
        downstream classification).
    """

    DEFAULT_HF_REPO = "MahmoodLab/TITAN"

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cuda",
        no_proj: bool = True,
        **build_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.no_proj = no_proj
        self.enc_name = "titan"

        if sys.version_info < (3, 10):
            raise RuntimeError("TITAN requires Python ≥ 3.10.")

        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "[TITAN] `transformers` is required to load the HuggingFace model."
            ) from e

        # ── Load the HF model ───────────────────────────────────────────
        if self.model_dir is None:
            repo = os.environ.get("TITAN_HF_REPO", self.DEFAULT_HF_REPO)
            revision = os.environ.get("TITAN_REVISION", None)
            cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
            token = os.environ.get("HF_TOKEN")
            print(f"[TITAN] Loading from HF Hub: {repo}")
            self.model = AutoModel.from_pretrained(
                repo,
                revision=revision,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=token,
                low_cpu_mem_usage=False,
            )
        else:
            print(f"[TITAN] Loading from local dir: {self.model_dir}")
            sys.path.insert(0, str(self.model_dir))
            self.model = AutoModel.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=str(self.model_dir),
                low_cpu_mem_usage=False,
            )

        # Resolve the slide-embedding dim from the HF config when possible;
        # fall back to 768 (TITAN-base) otherwise.
        embedding_dim = 768
        if hasattr(self.model, "config") and hasattr(self.model.config, "vision_config"):
            embedding_dim = int(getattr(self.model.config.vision_config, "embed_dim", 768))
        self.embedding_dim = int(os.environ.get("TITAN_EMBED_DIM", embedding_dim))

        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "features" not in batch or "coords" not in batch or "patch_size_lv0" not in batch:
            raise KeyError(
                "[TITAN] batch must contain 'features', 'coords', 'patch_size_lv0'."
            )

        feats = batch["features"]
        coords = batch["coords"]
        patch_size_lv0 = batch["patch_size_lv0"]
        if isinstance(patch_size_lv0, torch.Tensor):
            patch_size_lv0 = int(patch_size_lv0.item())
        else:
            patch_size_lv0 = int(patch_size_lv0)

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        feats = feats.to(self.device)
        coords = coords.to(self.device)

        # Preferred path: HF checkpoint exposes encode_slide_from_patch_features.
        if hasattr(self.model, "encode_slide_from_patch_features"):
            slide_emb = self.model.encode_slide_from_patch_features(
                patch_features=feats,
                patch_coords=coords,
                patch_size_lv0=patch_size_lv0,
            )
        else:
            # Fallback: call the internal vision encoder directly.
            if not hasattr(self.model, "vision_encoder"):
                raise RuntimeError(
                    "[TITAN] checkpoint exposes neither "
                    "`encode_slide_from_patch_features` nor `vision_encoder`."
                )
            try:
                slide_emb = self.model.vision_encoder(
                    feats, coords, patch_size_lv0, no_proj=self.no_proj,
                )
            except TypeError:
                slide_emb = self.model.vision_encoder(feats, coords, patch_size_lv0)

        if slide_emb.dim() == 1:
            slide_emb = slide_emb.unsqueeze(0)
        return slide_emb

    def forward_slide(
        self,
        patch_features: torch.Tensor,
        patch_coords: torch.Tensor,
        patch_size_lv0: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward({
            "features": patch_features,
            "coords": patch_coords,
            "patch_size_lv0": patch_size_lv0,
            **kwargs,
        })

    @property
    def slide_feature_dim(self) -> int:
        return self.embedding_dim

    @property
    def feature_dim(self) -> int:
        return self.embedding_dim
