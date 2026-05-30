from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# Optional: local checkout of prov-gigapath/prov-gigapath. When unset we rely
# on `pip install git+https://github.com/prov-gigapath/prov-gigapath` so
# `import gigapath` resolves through the regular site-packages path.
GIGAPATH_REPO = os.environ.get("GIGAPATH_REPO", "")
if GIGAPATH_REPO and GIGAPATH_REPO not in sys.path:
    sys.path.insert(0, GIGAPATH_REPO)


def _patch_longnet_segment_length() -> None:
    from gigapath.slide_encoder import LongNetViT

    def _get_optimal_segment_length(
        self,
        max_wsi_size: int = 262144,
        tile_size: int = 256,
    ) -> str:
        max_seq_len = (max_wsi_size // tile_size) ** 2
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        return str([int(x) for x in segment_length])

    LongNetViT.get_optimal_segment_length = _get_optimal_segment_length


class GigaPathSlideEncoder(nn.Module):
    DEFAULT_HF_REPO = "prov-gigapath/prov-gigapath"
    DEFAULT_MODEL_ARCH = "gigapath_slide_enc12l768d"
    TILE_EMBEDDING_DIM = 1536
    SLIDE_EMBEDDING_DIM = 768

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cuda",
        freeze: bool = True,
        global_pool: bool = False,
        **build_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.enc_name = "gigapath_slide"
        self.embedding_dim = self.SLIDE_EMBEDDING_DIM
        self.autocast_dtype = torch.float16 if self.device.type == "cuda" else None

        _patch_longnet_segment_length()

        from gigapath.slide_encoder import create_model as gigapath_create_model

        model_arch = build_kwargs.pop("model_arch", self.DEFAULT_MODEL_ARCH)
        self.model = gigapath_create_model(
            self._resolve_pretrained_source(),
            model_arch,
            self.TILE_EMBEDDING_DIM,
            global_pool=global_pool,
            **build_kwargs,
        )
        self.model.to(self.device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _resolve_pretrained_source(self) -> str:
        local_ckpt = os.environ.get("GIGAPATH_SLIDE_CKPT")
        if self.model_dir is not None:
            ckpt_path = self.model_dir / "slide_encoder.pth"
            if ckpt_path.exists():
                local_ckpt = str(ckpt_path)

        if local_ckpt and os.path.exists(local_ckpt):
            print(f"[GigaPath Slide] Loading from local: {local_ckpt}")
            return local_ckpt

        repo_id = os.environ.get("GIGAPATH_HF_REPO", self.DEFAULT_HF_REPO)
        print(f"[GigaPath Slide] Loading from HF Hub: {repo_id}")
        return f"hf_hub:{repo_id}"

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor | list[torch.Tensor]:
        if "features" not in batch:
            raise KeyError("[GigaPath] batch must contain key 'features'")
        if "coords" not in batch:
            raise KeyError("[GigaPath] batch must contain key 'coords'")

        patch_features = batch["features"]
        patch_coords = batch["coords"]
        all_layer_embed = bool(batch.get("all_layer_embed", False))

        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
        if patch_coords.dim() == 2:
            patch_coords = patch_coords.unsqueeze(0)

        patch_features = patch_features.to(self.device, dtype=torch.float32)
        patch_coords = patch_coords.to(self.device)

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(
                    patch_features,
                    patch_coords,
                    all_layer_embed=all_layer_embed,
                )
        else:
            outputs = self.model(
                patch_features,
                patch_coords,
                all_layer_embed=all_layer_embed,
            )
        return outputs if all_layer_embed else outputs[-1]

    def forward_slide(
        self,
        patch_features: torch.Tensor,
        patch_coords: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        kwargs.pop("tile_size", None)
        return self.forward(
            {
                "features": patch_features,
                "coords": patch_coords,
                **kwargs,
            }
        )

    @property
    def slide_feature_dim(self) -> int:
        return self.embedding_dim

    @property
    def feature_dim(self) -> int:
        return self.embedding_dim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prov-GigaPath slide encoder")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    encoder = GigaPathSlideEncoder(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        device=args.device,
    )
    print(f"GigaPath slide encoder: dim={encoder.embedding_dim}")

    if args.test:
        dummy_feats = torch.randn(100, 1536)
        dummy_coords = torch.randint(0, 50000, (100, 2)).float()
        out = encoder.forward_slide(dummy_feats, dummy_coords)
        print(f"Test output shape: {out.shape}")
