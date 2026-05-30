"""Distillation projector: maps student dimensions to teacher dimensions."""

import torch
import torch.nn as nn


class DistillationProjector(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        """
        Args:
            student_dim:   Student embed_dim (e.g. 384)
            teacher_dim:   Teacher embed_dim (e.g. 1536)
        """
        super().__init__()
        self.feat_projector = nn.Linear(student_dim, teacher_dim)
        self.pool_projector = nn.Linear(student_dim, teacher_dim)

    def project_tokens(self, student_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 256, S_dim) -> (B, 256, T_dim)"""
        return self.feat_projector(student_tokens)

    def project_pool(self, student_pool: torch.Tensor) -> torch.Tensor:
        """(B, S_dim) -> (B, T_dim)"""
        return self.pool_projector(student_pool)


def load_projector_from_checkpoint(checkpoint_path: str, map_location="cpu") -> DistillationProjector:
    """Load a DistillationProjector from a distillation checkpoint.

    The checkpoint must have been saved with --save_projector (the default).

    Usage:
        projector = load_projector_from_checkpoint("distill_step_100000.pt")
        projected = projector.project_tokens(student_tokens)
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "projector_state" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain projector weights. "
            "Was it saved with --no_save_projector?"
        )
    state = ckpt["projector_state"]
    # feat_projector.weight has shape (teacher_dim, student_dim)
    teacher_dim, student_dim = state["feat_projector.weight"].shape
    projector = DistillationProjector(student_dim, teacher_dim)
    projector.load_state_dict(state)
    return projector
