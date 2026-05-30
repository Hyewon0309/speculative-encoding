"""Self-contained PRISM slide-encoder wrapper.

PRISM (Paige.AI, Shaikovski et al. 2024) consumes 2560-d Virchow tile
embeddings and produces a single slide-level embedding via its
``slide_representations`` head. The official HuggingFace checkpoint
(``paige-ai/Prism``) ships its own modeling code, so this wrapper only:

    1. ``AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)``
    2. ``model.slide_representations(features)["image_embedding"]``

Mirrors the public surface of ``model/titan.py`` and ``model/gigapath.py`` so
``evaluator/runners/prism_subsample.py`` can stay structurally parallel to
the other slide-encoder runners.

Inputs
------
``features`` : ``(N, 2560)`` or ``(B, N, 2560)`` Virchow tile embeddings.

Output
------
``slide_embedding`` : ``(B, 1280)`` PRISM slide embedding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class PRISM(nn.Module):
    """PRISM slide encoder wrapper.

    Parameters
    ----------
    model_dir
        Optional local directory holding a PRISM checkpoint. When ``None``
        (the default), the model is fetched from HuggingFace using the
        repository identified by ``$PRISM_HF_REPO`` (falls back to
        ``paige-ai/Prism``).
    device
        Torch device for inference (``"cuda"`` / ``"cuda:0"`` / ``"cpu"``).
    use_amp
        Run ``slide_representations`` under ``torch.autocast(fp16)`` on CUDA
        for speed / VRAM. Enabled by default.
    """

    DEFAULT_HF_REPO = "paige-ai/Prism"
    SLIDE_EMBEDDING_DIM = 1280
    TILE_EMBEDDING_DIM = 2560

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cuda",
        use_amp: bool = True,
        **build_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.enc_name = "prism"
        self.embedding_dim = self.SLIDE_EMBEDDING_DIM

        if sys.version_info < (3, 10):
            raise RuntimeError("PRISM requires Python ≥ 3.10.")

        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "[PRISM] `transformers` is required to load the HuggingFace model."
            ) from e

        # ── Load the HF model ───────────────────────────────────────────
        if self.model_dir is None:
            repo = os.environ.get("PRISM_HF_REPO", self.DEFAULT_HF_REPO)
            revision = os.environ.get("PRISM_REVISION", None)
            cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
            token = os.environ.get("HF_TOKEN")
            print(f"[PRISM] Loading from HF Hub: {repo}")
            self.model = AutoModel.from_pretrained(
                repo,
                revision=revision,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=token,
            )
        else:
            print(f"[PRISM] Loading from local dir: {self.model_dir}")
            sys.path.insert(0, str(self.model_dir))
            self.model = AutoModel.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=str(self.model_dir),
            )

        # We do not use the text decoder for slide-level classification.
        if hasattr(self.model, "text_decoder"):
            self.model.text_decoder = None

        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "features" not in batch:
            raise KeyError("[PRISM] batch must contain key 'features'.")

        feats = batch["features"]
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        feats = feats.to(self.device, dtype=torch.float32)

        if self.use_amp:
            with torch.autocast("cuda", torch.float16):
                reprs = self.model.slide_representations(feats)
        else:
            reprs = self.model.slide_representations(feats)
        return reprs["image_embedding"]

    def forward_slide(self, patch_features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Convenience entry: ``patch_features`` of shape ``(N, 2560)`` or
        ``(B, N, 2560)`` → ``(B, 1280)`` slide embedding."""
        return self.forward({"features": patch_features, **kwargs})

    @property
    def slide_feature_dim(self) -> int:
        return self.embedding_dim

    @property
    def feature_dim(self) -> int:
        return self.embedding_dim
