"""Student model construction and checkpoint loading."""

import torch
import torch.nn as nn

# timm.layers (timm ≥ 1.0) was timm.models.layers in older versions.
try:
    import timm.layers as _timm_layers
except ModuleNotFoundError:
    import timm.models.layers as _timm_layers
from timm.models.vision_transformer import VisionTransformer

# Maps config string -> kwargs passed to VisionTransformer. ``swiglu`` is
# resolved lazily because ``SwiGLUPacked`` only exists in timm ≥ 0.9 and we
# want unrelated paths (gelu / silu) to keep working on older timm builds.
def _swiglu_kwargs() -> dict:
    if not hasattr(_timm_layers, "SwiGLUPacked"):
        raise RuntimeError(
            "act='swiglu' requires timm ≥ 0.9 (which exposes SwiGLUPacked). "
            "Upgrade timm or pick act='gelu'/'silu'."
        )
    return {"act_layer": nn.SiLU, "mlp_layer": _timm_layers.SwiGLUPacked}


_ACT_REGISTRY = {
    "gelu": {"act_layer": nn.GELU},
    "silu": {"act_layer": nn.SiLU},
}


def _resolve_act_kwargs(student_config: dict) -> dict:
    """Return a copy of *student_config* with the string ``act_layer`` key
    resolved to the actual PyTorch layer objects that timm expects.

    Backward-compatible: configs that lack ``act_layer`` are treated as
    ``"gelu"`` (the timm default), so old checkpoints keep working.
    """
    cfg = dict(student_config)
    act_name = cfg.pop("act_layer", "gelu")
    if act_name == "swiglu":
        act_kwargs = _swiglu_kwargs()
    else:
        act_kwargs = dict(_ACT_REGISTRY[act_name])
    cfg.update(act_kwargs)
    return cfg


def build_student_config(args) -> dict:
    """Build the student VisionTransformer config from CLI args."""
    return {
        "img_size": getattr(args, "student_img_size", None) or 224,
        "patch_size": getattr(args, "student_patch_size", None) or 14,
        "embed_dim": args.student_dim,
        "depth": args.student_depth,
        "num_heads": args.student_heads,
        "mlp_ratio": args.student_mlp_ratio,
        "num_classes": 0,
        "act_layer": args.student_act,
    }


def load_student(student_config: dict) -> VisionTransformer:
    """Create a student VisionTransformer from a config dict."""
    student = VisionTransformer(**_resolve_act_kwargs(student_config))
    return student


def _extract_student_state(ckpt):
    if "student_state" not in ckpt:
        raise KeyError("Checkpoint does not contain 'student_state'.")
    return {
        k.removeprefix("_orig_mod.").removeprefix("encoder."): v
        for k, v in ckpt["student_state"].items()
    }


def initialize_student_from_checkpoint(
    student: VisionTransformer,
    checkpoint_path: str,
    map_location="cpu",
):
    """Initialize an existing student model from a distillation checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    student.load_state_dict(_extract_student_state(ckpt), strict=True)


def load_student_from_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
    return_config: bool = False,
):
    """Load a distilled student encoder from a checkpoint file.

    Usage:
        student = load_student_from_checkpoint("distill_step_100000.pt")
        student.eval()
        features = student.forward_features(pixel_values)
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    student_config = ckpt["student_config"]
    student = load_student(student_config)
    student.load_state_dict(_extract_student_state(ckpt))
    if return_config:
        return student, dict(student_config)
    return student
