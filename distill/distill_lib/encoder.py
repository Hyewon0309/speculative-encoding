"""Encoder wrapper and output-extraction utilities."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class ImageEncoderWrapper(nn.Module):
    """Uniform calling convention: forward() -> forward_features()."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        if hasattr(self.encoder, "forward_features"):
            return self.encoder.forward_features(pixel_values)
        return self.encoder(pixel_values)


class GPUPixelNormalize(nn.Module):
    """Convert uint8 ``(B, 3, H, W)`` frames to normalized float32 on device.

    The distillation dataloader emits uint8 tensors so that pin-memory and
    PCIe bandwidth scale with the raw byte count instead of 4×-ed float32.
    This module is registered on the device inside the training loop and
    handles the ``/255`` + ``(x - mean) / std`` step in a single fused pass.

    Call exactly once per raw uint8 batch; the linear-probe path continues to
    use its own pre-normalized PIL pipeline so this module is not applied
    there.
    """

    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.tensor(tuple(mean), dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(tuple(std), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.to(torch.float32) / 255.0
        elif pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        return (pixel_values - self.mean) / self.std


def _extract_sequence(outputs):
    """Convert encoder outputs -> (B, seq_len, dim) tensor."""
    if isinstance(outputs, dict):
        for key in (
            "x_norm_patchtokens",
            "last_hidden_state",
            "patch_tokens",
            "features",
            "pooler_output",
        ):
            if key in outputs:
                outputs = outputs[key]
                break
    elif hasattr(outputs, "last_hidden_state"):
        outputs = outputs.last_hidden_state
    elif hasattr(outputs, "pooler_output"):
        outputs = outputs.pooler_output
    elif isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    if not isinstance(outputs, torch.Tensor):
        raise ValueError(f"Unsupported encoder output type: {type(outputs)}")

    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(1)
    elif outputs.dim() == 4:
        outputs = outputs.flatten(2).transpose(1, 2)
    return outputs


def _infer_image_prefix_tokens(encoder: nn.Module) -> int:
    """Auto-detect number of CLS + register tokens from a timm model."""
    # Unwrap ImageEncoderWrapper if present
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder

    num_prefix_tokens = getattr(encoder, "num_prefix_tokens", None)
    if num_prefix_tokens is not None:
        try:
            return int(num_prefix_tokens)
        except (TypeError, ValueError):
            pass
    num_reg_tokens = getattr(encoder, "num_reg_tokens", None)
    if num_reg_tokens is not None:
        try:
            num_reg_tokens = int(num_reg_tokens)
        except (TypeError, ValueError):
            num_reg_tokens = 0
        has_cls = hasattr(encoder, "cls_token") or hasattr(encoder, "class_embedding")
        return num_reg_tokens + (1 if has_cls else 0)
    if hasattr(encoder, "cls_token") or hasattr(encoder, "class_embedding"):
        return 1
    return 0


def _drop_prefix_tokens(embeddings: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:
    """Remove CLS + register tokens from (B, seq_len, dim)."""
    if num_prefix_tokens and num_prefix_tokens > 0 and embeddings.size(1) > num_prefix_tokens:
        return embeddings[:, num_prefix_tokens:, :]
    return embeddings


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    if isinstance(module, DDP):
        return module.module
    return module
