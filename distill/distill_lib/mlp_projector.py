"""Standalone MLP projector that maps student features to teacher features.

Trained with both teacher and student frozen — the projector's job is to learn
an ``in_dim → out_dim`` map that makes the student's embedding look like the
teacher's under an MSE objective.

Checkpoints carry their own architecture config so ``load_mlp_projector`` can
reconstruct the module without knowing the original arguments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class MLPProjector(nn.Module):
    """``in_dim → out_dim`` MLP with configurable depth and activation.

    Layout (``num_hidden_layers=H``):
        Linear(in_dim, hidden)
        Activation
        [Linear(hidden, hidden), Activation] × (H - 1)
        Linear(hidden, out_dim)

    ``num_hidden_layers=0`` degenerates to a single linear map.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        activation = activation.lower()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from {sorted(_ACTIVATIONS)}."
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        if num_hidden_layers <= 0:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_cls())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "activation": self.activation,
        }


def save_mlp_projector(
    projector: MLPProjector,
    path: str,
    *,
    meta: dict | None = None,
) -> None:
    """Write a self-contained MLP checkpoint.

    The checkpoint contains:
      * ``mlp_config`` — kwargs to rebuild the module
      * ``mlp_state`` — ``state_dict`` with ``_orig_mod.`` / ``module.`` prefixes stripped
      * ``meta`` — free-form dict (teacher/student info, step, loss, etc.)
    """
    state_dict = projector.state_dict()
    cleaned = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state_dict.items()
    }
    payload = {
        "mlp_config": projector.config(),
        "mlp_state": cleaned,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_mlp_projector(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> MLPProjector:
    """Rebuild an ``MLPProjector`` from a checkpoint written by ``save_mlp_projector``.

    Usage::

        from distill_lib.mlp_projector import load_mlp_projector
        projector = load_mlp_projector("mlp_projector.pt").eval()
        teacher_feat = projector(student_cls)  # (B, 384) -> (B, 768)
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "mlp_config" not in ckpt or "mlp_state" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} is missing mlp_config or mlp_state."
        )
    projector = MLPProjector(**ckpt["mlp_config"])
    projector.load_state_dict(ckpt["mlp_state"], strict=True)
    return projector
