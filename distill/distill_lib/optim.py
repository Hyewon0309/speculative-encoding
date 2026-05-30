"""Optimizer builders for distillation training."""

import inspect

import torch


def _trainable_named_params(module):
    if module is None:
        return []
    return [(name, param) for name, param in module.named_parameters() if param.requires_grad]


def _split_muon_params(named_params):
    muon_params = []
    adamw_params = []
    for _, param in named_params:
        # torch.optim.Muon only accepts exactly 2D tensors.
        # ViT modules also have 3D parameters such as cls/register tokens and
        # 4D patch-embed conv kernels, which must stay on the AdamW fallback.
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def _flatten_params(param_groups):
    params = []
    for group in param_groups:
        group_params = group.get("params", [])
        if isinstance(group_params, torch.Tensor):
            params.append(group_params)
        else:
            params.extend(group_params)
    return params


def adamw_supports_fused() -> bool:
    return "fused" in inspect.signature(torch.optim.AdamW).parameters


def can_use_fused_adamw(param_groups) -> bool:
    if not adamw_supports_fused():
        return False
    params = _flatten_params(param_groups)
    if not params:
        return False
    return all(
        param.is_cuda and torch.is_floating_point(param)
        for param in params
    )


def build_optimizers(args, student, projector, auxiliary_modules=None):
    """Return a list of ``(name, optimizer)`` tuples and the clip parameter list."""
    auxiliary_modules = auxiliary_modules or []
    student_named = _trainable_named_params(student)
    projector_named = _trainable_named_params(projector)
    auxiliary_named = []
    for module in auxiliary_modules:
        auxiliary_named.extend(_trainable_named_params(module))
    clip_params = [param for _, param in student_named + projector_named + auxiliary_named]
    proj_lr = args.projector_lr if args.projector_lr is not None else args.lr

    if args.optimizer in {"adamw", "adamw8bit"}:
        param_groups = []
        if student_named:
            param_groups.append({"params": [p for _, p in student_named], "lr": args.lr})
        if projector_named:
            param_groups.append({"params": [p for _, p in projector_named], "lr": proj_lr})
        if auxiliary_named:
            param_groups.append({"params": [p for _, p in auxiliary_named], "lr": proj_lr})

        if args.optimizer == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise RuntimeError(
                    "--optimizer adamw8bit requires bitsandbytes to be installed."
                ) from exc
            optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=args.weight_decay)
            return [("adamw8bit", optimizer)], clip_params

        adamw_kwargs = {"weight_decay": args.weight_decay}
        if can_use_fused_adamw(param_groups):
            adamw_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
        optimizer_name = "adamw_fused" if adamw_kwargs.get("fused") else "adamw"
        return [(optimizer_name, optimizer)], clip_params

    if args.optimizer != "muon":
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")

    muon_params, student_adamw_params = _split_muon_params(student_named)
    optimizers = []

    if muon_params:
        optimizers.append((
            "muon",
            torch.optim.Muon(
                muon_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.muon_momentum,
                nesterov=args.muon_nesterov,
                eps=args.muon_eps,
                ns_steps=args.muon_ns_steps,
            ),
        ))

    adamw_groups = []
    if student_adamw_params:
        adamw_groups.append({"params": student_adamw_params, "lr": args.lr})
    if projector_named:
        adamw_groups.append({"params": [p for _, p in projector_named], "lr": proj_lr})
    if auxiliary_named:
        adamw_groups.append({"params": [p for _, p in auxiliary_named], "lr": proj_lr})
    if adamw_groups:
        adamw_kwargs = {"weight_decay": args.weight_decay}
        if can_use_fused_adamw(adamw_groups):
            adamw_kwargs["fused"] = True
        optimizer_name = "adamw_fused" if adamw_kwargs.get("fused") else "adamw"
        optimizers.append((
            optimizer_name,
            torch.optim.AdamW(adamw_groups, **adamw_kwargs),
        ))

    return optimizers, clip_params
