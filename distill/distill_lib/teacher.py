"""Teacher model loading with multi-model support."""

import json
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2

from distill_lib.encoder import ImageEncoderWrapper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_random_crop_transform(img_size, mean, std):
    """Distillation-time transform that keeps tensors as uint8.

    Fixed-size ``RandomCrop(img_size)`` — no scale/zoom randomization. Every
    output is exactly ``img_size × img_size``, matching the teacher's native
    input resolution. ``pad_if_needed`` keeps the crop well-defined when the
    source image is smaller than ``img_size``.

    Crop + flip stay in uint8 to minimise CPU memory traffic. The dtype
    upcast and per-channel normalization happen on the GPU in the training
    loop (see ``Distiller._normalize_pixels``), which also shrinks the
    pin-memory/PCIe transfer by ~4× compared with sending float32 frames.

    The ``mean`` / ``std`` arguments are accepted for API compatibility but
    are not consumed here — they travel via ``teacher_meta`` and end up on
    the GPU normalizer.
    """
    del mean, std
    return transforms_v2.Compose(
        [
            transforms_v2.RandomCrop(img_size, pad_if_needed=True),
            transforms_v2.RandomHorizontalFlip(),
        ]
    )


# ---------------------------------------------------------------------------
# Wrapper: HuggingFace transformers vision models (PLIP, MedSiglip)
# ---------------------------------------------------------------------------

class _HFVisionEncoder(nn.Module):
    """Wraps a HuggingFace ``vision_model`` so that ``forward_features()``
    returns the full token sequence ``(B, seq_len, dim)``."""

    def __init__(self, vision_model: nn.Module, num_prefix_tokens: int):
        super().__init__()
        self.vision_model = vision_model
        self.num_prefix_tokens = num_prefix_tokens

    def forward_features(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    def forward(self, pixel_values):
        return self.forward_features(pixel_values)


# ---------------------------------------------------------------------------
# Wrapper: open_clip visual encoder (BiomedCLIP, OpenAI CLIP)
# ---------------------------------------------------------------------------

class _OpenCLIPSequenceEncoder(nn.Module):
    """Wraps an ``open_clip`` ``model.visual`` (VisionTransformer) so that
    ``forward_features()`` returns the full token sequence ``(B, seq_len, dim)``
    **before** pooling/projection.

    The forward path mirrors open_clip's ``VisionTransformer._intermediate_layers``
    but returns the post-LN sequence instead of just the pooled vector.
    """

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual
        self.num_prefix_tokens = 1  # CLS token

    def forward_features(self, pixel_values):
        v = self.visual
        x = v.conv1(pixel_values)  # patch embed
        # x: (B, embed_dim, grid_h, grid_w) → (B, num_patches, embed_dim)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # Prepend CLS token
        cls_token = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)

        # Positional embedding
        x = x + v.positional_embedding.to(x.dtype)

        # Pre-LN (if present)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Post-LN — return full sequence (B, seq_len, dim)
        x = v.ln_post(x)
        return x

    def forward(self, pixel_values):
        return self.forward_features(pixel_values)


_HF_REQUIRED_FILES = ("config.json",)
_HF_WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def _snapshot_is_complete(snapshot_dir: Path) -> bool:
    """A snapshot is usable only if it has the config plus a weights file."""
    if not snapshot_dir.is_dir():
        return False
    for required in _HF_REQUIRED_FILES:
        if not (snapshot_dir / required).is_file():
            return False
    return any((snapshot_dir / w).is_file() for w in _HF_WEIGHT_CANDIDATES)


def _candidate_hf_cache_dirs() -> list:
    """Ordered list of HF cache dirs to search, de-duplicated.

    Covers both the env-configured cache and the common defaults so that a
    partial snapshot left behind in one location (e.g. from a gated-repo
    403 where only README.md was public) doesn't hide a complete snapshot
    sitting in another.
    """
    candidates = []

    def _add(path):
        if not path:
            return
        resolved = os.path.abspath(os.path.expanduser(str(path)))
        if resolved not in candidates:
            candidates.append(resolved)

    _add(os.environ.get("HF_HUB_CACHE"))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        _add(os.path.join(hf_home, "hub"))
    _add(os.path.expanduser("~/.cache/huggingface/hub"))
    return candidates


def _repo_folder_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _scan_cached_snapshot(repo_id: str) -> Path:
    """Return the first locally cached snapshot directory with config + weights.

    Handles the case where ``HF_HOME`` points at an incomplete cache (e.g.
    a README-only 403 fallback) while a sibling cache directory actually
    holds the full snapshot.
    """
    folder = _repo_folder_name(repo_id)
    for cache_dir in _candidate_hf_cache_dirs():
        repo_dir = Path(cache_dir) / folder
        snapshots_dir = repo_dir / "snapshots"
        refs_main = repo_dir / "refs" / "main"

        ordered_snapshots = []
        if refs_main.is_file():
            commit_hash = refs_main.read_text().strip()
            preferred = snapshots_dir / commit_hash
            if preferred.is_dir():
                ordered_snapshots.append(preferred)
        if snapshots_dir.is_dir():
            for entry in snapshots_dir.iterdir():
                if entry.is_dir() and entry not in ordered_snapshots:
                    ordered_snapshots.append(entry)

        for snapshot in ordered_snapshots:
            if _snapshot_is_complete(snapshot):
                return snapshot
    raise FileNotFoundError(
        f"No complete local HF snapshot for {repo_id!r} was found in any of: "
        f"{_candidate_hf_cache_dirs()}"
    )


def _resolve_hf_snapshot(repo_id: str) -> Path:
    """Return a local snapshot directory for ``repo_id`` that has config + weights.

    Resolution order:
      1. Scan all candidate HF cache dirs for a **complete** snapshot (has
         config.json + a weights file). This is the primary path and is
         robust to an incomplete cache left by a failed online fetch.
      2. Fall back to ``snapshot_download(local_files_only=True)``.
      3. Finally, attempt an online download via ``snapshot_download``
         (requires HF_TOKEN for gated repos such as paige-ai/Virchow and
         prov-gigapath/prov-gigapath).
    """
    from huggingface_hub import snapshot_download

    try:
        return _scan_cached_snapshot(repo_id)
    except FileNotFoundError:
        pass

    cache_dir = os.environ.get("HF_HUB_CACHE")
    if cache_dir is None and os.environ.get("HF_HOME"):
        cache_dir = os.path.join(os.environ["HF_HOME"], "hub")

    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id, local_files_only=True, cache_dir=cache_dir,
        ))
        if _snapshot_is_complete(snapshot):
            return snapshot
    except Exception:
        pass

    token = os.environ.get("HF_TOKEN")
    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id, cache_dir=cache_dir, token=token,
        ))
        if _snapshot_is_complete(snapshot):
            return snapshot
        raise FileNotFoundError(
            f"Downloaded snapshot for {repo_id!r} at {snapshot} is incomplete "
            f"(missing config.json or weights file)."
        )
    except Exception as online_err:
        raise RuntimeError(
            f"Could not locate a complete local HF snapshot for {repo_id!r} and "
            f"online download also failed.\n"
            f"  searched cache dirs: {_candidate_hf_cache_dirs()}\n"
            f"  online error: {type(online_err).__name__}: {online_err}\n"
            f"If the repo is gated, set HF_TOKEN; otherwise ensure a complete "
            f"snapshot exists under one of the cache dirs above."
        ) from online_err


def _load_timm_weights_from_snapshot(snapshot_dir: Path) -> dict:
    """Load a timm-style state_dict from a local HF snapshot directory.

    Prefers ``model.safetensors`` over ``pytorch_model.bin`` to avoid the
    slow pickle path. Strips any ``module.`` DDP prefix.
    """
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = snapshot_dir / name
        if candidate.is_file():
            weights_path = candidate
            break
    else:
        raise FileNotFoundError(
            f"No weights file (model.safetensors or pytorch_model.bin) found in {snapshot_dir}"
        )

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
    else:
        state_dict = torch.load(str(weights_path), map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_timm_from_hf_snapshot(repo_id: str, **extra_model_kwargs):
    """Build a ``timm`` model from a HF repo without going through
    ``timm.create_model('hf_hub:...')``.

    Reads ``config.json`` from the local snapshot, instantiates the
    architecture with ``pretrained=False``, loads weights manually, and
    attaches ``pretrained_cfg`` so downstream ``resolve_data_config``
    calls still work.
    """
    import json

    snapshot_dir = _resolve_hf_snapshot(repo_id)
    config_path = snapshot_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"HF snapshot for {repo_id!r} is missing config.json at {config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    arch = config["architecture"]
    model_args = dict(config.get("model_args") or {})
    pretrained_cfg = dict(config.get("pretrained_cfg") or {})
    model_args.update(extra_model_kwargs)

    model = timm.create_model(arch, pretrained=False, **model_args)

    state_dict = _load_timm_weights_from_snapshot(snapshot_dir)
    model.load_state_dict(state_dict, strict=True)

    try:
        model.pretrained_cfg = pretrained_cfg
    except Exception:  # pragma: no cover - read-only attrs are unexpected here
        pass
    return model


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict)}")
    first_key = next(iter(state_dict), None)
    if first_key and first_key.startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_teacher_checkpoint_path(source, default_repo_id, candidate_filenames):
    """Resolve a teacher checkpoint from file/dir/HF repo id."""
    from huggingface_hub import hf_hub_download

    if source is None:
        repo_id = default_repo_id
    elif os.path.isfile(source):
        return source
    elif os.path.isdir(source):
        for filename in candidate_filenames:
            candidate = os.path.join(source, filename)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find any of {candidate_filenames} under directory: {source}"
        )
    elif source.startswith("hf_hub:"):
        repo_id = source[len("hf_hub:"):]
    else:
        source_path = Path(source).expanduser()
        if source_path.exists():
            if source_path.is_file():
                return str(source_path)
            if source_path.is_dir():
                for filename in candidate_filenames:
                    candidate = source_path / filename
                    if candidate.is_file():
                        return str(candidate)
                raise FileNotFoundError(
                    f"Could not find any of {candidate_filenames} under directory: {source}"
                )
        repo_id = source

    last_error = None
    for filename in candidate_filenames:
        try:
            return hf_hub_download(repo_id, filename=filename)
        except Exception as exc:  # pragma: no cover - exact HF error type varies
            last_error = exc
    raise RuntimeError(
        f"Failed to resolve checkpoint from Hugging Face repo '{repo_id}' "
        f"with candidate filenames {candidate_filenames}."
    ) from last_error


# ---------------------------------------------------------------------------
# Per-model loaders
# ---------------------------------------------------------------------------

def _load_uni_teacher(args):
    """Load UNI (timm ViT-H/14, 1536-dim)."""
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs,
    )
    preprocess_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    eval_preprocess = create_transform(**preprocess_cfg)

    input_size = preprocess_cfg.get("input_size", (3, 224, 224))
    crop_size = input_size[-1] if isinstance(input_size, (list, tuple)) else input_size
    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    distill_preprocess = _build_random_crop_transform(crop_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(model)
    teacher_meta = {
        "img_size": 224, "patch_size": 14, "embed_dim": 1536,
        "mean": mean, "std": std,
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_biomedclip_teacher(args):
    """Load BiomedCLIP via open_clip, wrapping visual with _OpenCLIPSequenceEncoder."""
    from open_clip import create_model_and_transforms
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

    model_path = args.teacher_model_path
    model_name = args.teacher_model_name or "biomedclip_local"

    config_path = os.path.join(model_path, "open_clip_config.json")
    weights_path = os.path.join(model_path, "open_clip_pytorch_model.bin")

    with open(config_path, "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    model, _, eval_preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=weights_path,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )

    # Determine embed_dim from the visual trunk's LN
    embed_dim = model.visual.ln_post.normalized_shape[0]

    # Image / patch sizes from config
    vision_cfg = model_cfg.get("vision_cfg", {})
    img_size = vision_cfg.get("image_size", 224)
    patch_size = vision_cfg.get("patch_size", 16)

    # Normalization for distill transform
    mean = preprocess_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073))
    std = preprocess_cfg.get("std", (0.26862954, 0.26130258, 0.27577711))
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(_OpenCLIPSequenceEncoder(model.visual))
    teacher_meta = {
        "img_size": img_size, "patch_size": patch_size, "embed_dim": embed_dim,
        "mean": tuple(mean), "std": tuple(std),
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_openaiclip_teacher(args):
    """Load OpenAI CLIP via open_clip with pretrained='openai'."""
    from open_clip import create_model_and_transforms

    arch_name = args.teacher_model_path or "ViT-L-14"

    model, _, eval_preprocess = create_model_and_transforms(
        model_name=arch_name,
        pretrained="openai",
    )

    # Determine embed_dim from the visual trunk's LN
    embed_dim = model.visual.ln_post.normalized_shape[0]

    # Determine image/patch sizes from the visual model
    v = model.visual
    # open_clip stores image_size on the visual module
    img_size = getattr(v, "image_size", None)
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[0]
    if img_size is None:
        img_size = 224

    # Infer patch size from conv1 kernel
    patch_size = v.conv1.kernel_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]

    # OpenAI CLIP normalization
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(_OpenCLIPSequenceEncoder(model.visual))
    teacher_meta = {
        "img_size": img_size, "patch_size": patch_size, "embed_dim": embed_dim,
        "mean": tuple(mean), "std": tuple(std),
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_plip_teacher(args):
    """Load PLIP via transformers CLIPModel, wrapping vision_model."""
    from transformers import CLIPModel, CLIPProcessor

    model_path = args.teacher_model_path
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)

    vision_config = model.vision_model.config
    embed_dim = vision_config.hidden_size
    img_size = vision_config.image_size
    patch_size = vision_config.patch_size

    # Build eval transform from the processor
    eval_preprocess = processor.image_processor

    # Distill transform with CLIP normalization
    mean = tuple(eval_preprocess.image_mean)
    std = tuple(eval_preprocess.image_std)
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(
        _HFVisionEncoder(model.vision_model, num_prefix_tokens=1)
    )
    teacher_meta = {
        "img_size": img_size, "patch_size": patch_size, "embed_dim": embed_dim,
        "mean": tuple(mean), "std": tuple(std),
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_medsiglip_teacher(args):
    """Load MedSiglip via transformers AutoModel."""
    from transformers import AutoModel, AutoProcessor

    model_id = args.teacher_model_path or "google/medsiglip-448"
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    vision_config = model.vision_model.config
    embed_dim = vision_config.hidden_size
    img_size = vision_config.image_size
    patch_size = vision_config.patch_size

    # Build eval transform from the processor
    eval_preprocess = processor.image_processor

    # Distill transform with Siglip normalization
    mean = tuple(eval_preprocess.image_mean)
    std = tuple(eval_preprocess.image_std)
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Siglip has no CLS token
    encoder = ImageEncoderWrapper(
        _HFVisionEncoder(model.vision_model, num_prefix_tokens=0)
    )
    teacher_meta = {
        "img_size": img_size, "patch_size": patch_size, "embed_dim": embed_dim,
        "mean": tuple(mean), "std": tuple(std),
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_virchow_teacher(args):
    """Load Virchow tile encoder (paige-ai/Virchow).

    ViT-H/14, SwiGLUPacked MLP + SiLU. ``forward_features`` returns
    ``(B, 1+N, 1280)`` (CLS + patch tokens). The distillation pipeline drops
    the single prefix token via ``num_prefix_tokens=1``.
    """
    from timm.layers import SwiGLUPacked

    repo_id = args.teacher_model_path or os.environ.get(
        "VIRCHOW_HF_REPO", "paige-ai/Virchow"
    )
    # Strip any "hf-hub:" / "hf_hub:" prefix — we resolve the snapshot locally
    # so timm never makes a (potentially 403-gated) network call.
    for prefix in ("hf-hub:", "hf_hub:"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix):]
            break

    model = _build_timm_from_hf_snapshot(
        repo_id,
        num_classes=0,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
    )
    preprocess_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    eval_preprocess = create_transform(**preprocess_cfg)

    input_size = preprocess_cfg.get("input_size", (3, 224, 224))
    crop_size = input_size[-1] if isinstance(input_size, (list, tuple)) else input_size
    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    distill_preprocess = _build_random_crop_transform(crop_size, mean, std)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(model)
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    teacher_meta = {
        "img_size": crop_size,
        "patch_size": patch_size,
        "embed_dim": model.embed_dim,
        "mean": mean,
        "std": std,
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_provgigapath_teacher(args):
    """Load Prov-GigaPath tile encoder (prov-gigapath/prov-gigapath).

    ViT-giant/16, 1536-dim. ``forward_features`` returns ``(B, 1+N, 1536)``.
    Eval preprocess follows the official recipe: ``Resize(256, BICUBIC) ->
    CenterCrop(224) -> ToTensor -> Normalize(ImageNet)``.
    """
    repo_id = args.teacher_model_path or os.environ.get(
        "PROVGIGAPATH_REPO_ID", "prov-gigapath/prov-gigapath"
    )
    for prefix in ("hf-hub:", "hf_hub:"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix):]
            break

    model = _build_timm_from_hf_snapshot(repo_id)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_size = 224
    crop_source_size = 256
    eval_preprocess = transforms.Compose(
        [
            transforms.Resize(crop_source_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # Distillation-time augmentation mirrors the official eval geometry:
    # take a 256-sized random window, then resize to 224 (BICUBIC). Keeps
    # tensors in uint8 so the GPU normalizer handles dtype + normalization.
    distill_preprocess = transforms_v2.Compose(
        [
            transforms_v2.RandomCrop(crop_source_size, pad_if_needed=True),
            transforms_v2.Resize(
                img_size,
                interpolation=transforms_v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms_v2.RandomHorizontalFlip(),
        ]
    )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(model)
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    teacher_meta = {
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": model.embed_dim,
        "mean": mean,
        "std": std,
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


def _load_conch_teacher(args):
    """Load CONCH (CoCa ViT-B/16) via conch.open_clip_custom.

    The trunk is a timm VisionTransformer — we wrap it directly with
    ImageEncoderWrapper, bypassing the CoCa attentional poolers / projectors.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    model_name = args.teacher_model_name or "conch_ViT-B-16"
    checkpoint = args.teacher_model_path or "hf_hub:MahmoodLab/conch"
    model, eval_preprocess = create_model_from_pretrained(model_name, checkpoint)

    trunk = model.visual.trunk  # timm VisionTransformer

    img_size = trunk.patch_embed.img_size
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[0]
    patch_size = trunk.patch_embed.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    embed_dim = trunk.embed_dim

    # Normalization — CONCH uses OpenAI CLIP values
    mean = getattr(model.visual, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(model.visual, "image_std", (0.26862954, 0.26130258, 0.27577711))
    mean = tuple(mean) if not isinstance(mean, tuple) else mean
    std = tuple(std) if not isinstance(std, tuple) else std
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    for param in trunk.parameters():
        param.requires_grad = False
    trunk.eval()

    # Wrap the trunk directly — it already has forward_features() and
    # num_prefix_tokens=1 (CLS token) as a timm attribute.
    encoder = ImageEncoderWrapper(trunk)
    teacher_meta = {
        "img_size": img_size, "patch_size": patch_size, "embed_dim": embed_dim,
        "mean": mean, "std": std,
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


class _ConchV15AttentionalPooler(nn.Module):
    """Single-query cross-attention pooler used by CONCH v1.5.

    Reimplements MahmoodLab/TITAN's ``AttentionalPooler`` without the einops
    dependency. Parameter names match the original (``query``, ``ln_k``,
    ``ln_q``, ``to_q``, ``to_kv``, ``to_out``) so the state dict loads directly.
    """

    def __init__(self, d_model: int = 768, context_dim: int = 1024,
                 n_head: int = 8, n_queries: int = 1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_queries = n_queries
        self.heads = n_head
        self.dim_head = d_model // n_head
        self.scale = self.dim_head ** -0.5

        self.query = nn.Parameter(torch.empty(n_queries, d_model))
        self.ln_k = nn.LayerNorm(context_dim)
        self.ln_q = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, d_model * 2, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, N, context_dim) → (B, n_queries, d_model)."""
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        x = self.ln_k(x)
        q = self.ln_q(self.query.unsqueeze(0).expand(B, -1, -1))
        q = self.to_q(q)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = q.view(B, self.n_queries, h, dh).transpose(1, 2)
        k = k.view(B, N, h, dh).transpose(1, 2)
        v = v.view(B, N, h, dh).transpose(1, 2)

        q = q * self.scale
        sim = q @ k.transpose(-2, -1)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, self.n_queries, h * dh)
        return self.to_out(out)


class _ConchV15ProjectedEncoder(nn.Module):
    """CONCH v1.5 trunk + attentional pooler + ``ln_contrast``.

    ``forward_features`` returns the 768-dim contrast-projected pooled vector
    as a ``(B, 1, 768)`` sequence so the standard ``ImageEncoderWrapper`` /
    ``_extract_sequence`` path treats it like a single CLS-style token.
    """

    num_prefix_tokens = 1

    def __init__(self, trunk: nn.Module, attn_pool_contrast: nn.Module,
                 ln_contrast: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.attn_pool_contrast = attn_pool_contrast
        self.ln_contrast = ln_contrast

    @property
    def cls_token(self):
        return self.trunk.cls_token

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk.forward_features(pixel_values)  # (B, 1+N, 1024)
        pooled = self.attn_pool_contrast(tokens)            # (B, 1, 768)
        pooled = self.ln_contrast(pooled[:, 0])             # (B, 768)
        return pooled.unsqueeze(1)                          # (B, 1, 768)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(pixel_values)


def _load_conchv15_teacher(args):
    """Load CONCH v1.5 (ViT-L/16 trunk, optional 1024→768 contrast projection).

    When ``args.conchv15_use_projection`` is True (default) the teacher includes
    the single-query attentional pooler + ``ln_contrast`` and returns a
    ``(B, 1, 768)`` sequence. When False the teacher is the bare trunk and
    returns ``(B, 1+N, 1024)``.
    """
    from timm.layers import resample_abs_pos_embed

    use_projection = bool(getattr(args, "conchv15_use_projection", True))

    checkpoint_path = _resolve_teacher_checkpoint_path(
        source=args.teacher_model_path,
        default_repo_id="MahmoodLab/conchv1_5",
        candidate_filenames=("pytorch_model_vision.bin", "pytorch_model.bin"),
    )

    img_size = 448
    patch_size = 16
    trunk_embed_dim = 1024
    contrast_embed_dim = 768

    trunk = timm.models.vision_transformer.VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=trunk_embed_dim,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1.0,
        num_classes=0,
        dynamic_img_size=True,
    )

    attn_pool_contrast = None
    ln_contrast = None
    if use_projection:
        attn_pool_contrast = _ConchV15AttentionalPooler(
            d_model=contrast_embed_dim,
            context_dim=trunk_embed_dim,
            n_head=8,
            n_queries=1,
        )
        ln_contrast = nn.LayerNorm(contrast_embed_dim)

    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw_state_dict)

    trunk_keys = set(trunk.state_dict().keys())
    pooler_keys = set(attn_pool_contrast.state_dict().keys()) if use_projection else set()
    ln_keys = set(ln_contrast.state_dict().keys()) if use_projection else set()

    trunk_state: dict = {}
    pooler_state: dict = {}
    ln_state: dict = {}
    for key, value in state_dict.items():
        if key.startswith("trunk."):
            sub = key[len("trunk."):]
            if sub in trunk_keys:
                trunk_state[sub] = value
        elif use_projection and key.startswith("attn_pool_contrast."):
            sub = key[len("attn_pool_contrast."):]
            if sub in pooler_keys:
                pooler_state[sub] = value
        elif use_projection and key.startswith("ln_contrast."):
            sub = key[len("ln_contrast."):]
            if sub in ln_keys:
                ln_state[sub] = value
        elif key in trunk_keys:
            trunk_state[key] = value

    missing_trunk = trunk_keys - set(trunk_state.keys())
    if missing_trunk:
        raise RuntimeError(
            "CONCH v1.5 checkpoint is missing trunk parameters: "
            f"{sorted(missing_trunk)[:8]}"
        )
    if use_projection:
        missing_pooler = pooler_keys - set(pooler_state.keys())
        if missing_pooler:
            raise RuntimeError(
                "CONCH v1.5 checkpoint is missing contrast pooler parameters: "
                f"{sorted(missing_pooler)}"
            )
        missing_ln = ln_keys - set(ln_state.keys())
        if missing_ln:
            raise RuntimeError(
                "CONCH v1.5 checkpoint is missing ln_contrast parameters: "
                f"{sorted(missing_ln)}"
            )

    if trunk_state["pos_embed"].shape != trunk.pos_embed.shape:
        trunk_state["pos_embed"] = resample_abs_pos_embed(
            trunk_state["pos_embed"],
            new_size=trunk.patch_embed.grid_size,
            num_prefix_tokens=getattr(trunk, "num_prefix_tokens", 1),
            interpolation="bilinear",
            antialias=False,
            verbose=False,
        )

    trunk.load_state_dict(trunk_state, strict=True)
    if use_projection:
        attn_pool_contrast.load_state_dict(pooler_state, strict=True)
        ln_contrast.load_state_dict(ln_state, strict=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    eval_preprocess = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    distill_preprocess = _build_random_crop_transform(img_size, mean, std)

    if use_projection:
        model = _ConchV15ProjectedEncoder(trunk, attn_pool_contrast, ln_contrast)
        out_embed_dim = contrast_embed_dim
    else:
        model = trunk
        out_embed_dim = trunk_embed_dim
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = ImageEncoderWrapper(model)
    teacher_meta = {
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": out_embed_dim,
        "mean": mean,
        "std": std,
    }
    return encoder, eval_preprocess, distill_preprocess, teacher_meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEACHER_LOADERS = {
    "uni": _load_uni_teacher,
    "biomedclip": _load_biomedclip_teacher,
    "openaiclip": _load_openaiclip_teacher,
    "plip": _load_plip_teacher,
    "medsiglip": _load_medsiglip_teacher,
    "conch": _load_conch_teacher,
    "conchv15": _load_conchv15_teacher,
    "conch_v15": _load_conchv15_teacher,
    "conchv1_5": _load_conchv15_teacher,
    "virchow": _load_virchow_teacher,
    "prism": _load_virchow_teacher,
    "provgigapath": _load_provgigapath_teacher,
    "prov_gigapath": _load_provgigapath_teacher,
    "prov-gigapath": _load_provgigapath_teacher,
    "gigapath": _load_provgigapath_teacher,
}


def load_teacher(args):
    """Load teacher encoder + preprocesses.

    Returns (ImageEncoderWrapper, eval_preprocess, distill_preprocess, teacher_meta).
    ``teacher_meta`` is a dict with keys: img_size, patch_size, embed_dim.
    """
    name = args.teacher_model
    loader = _TEACHER_LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"Unsupported teacher model: {name}. "
            f"Choose from {list(_TEACHER_LOADERS.keys())}."
        )
    return loader(args)
