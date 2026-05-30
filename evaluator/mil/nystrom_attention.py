"""Nystrom attention (from PathGen architecture.nystrom_attention). Requires einops."""

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange, reduce
except ImportError:
    rearrange = reduce = None


def _rearrange(x, pattern, **kwargs):
    if rearrange is None:
        raise ImportError("mil with TransMIL/transformer needs einops: pip install einops")
    return rearrange(x, pattern, **kwargs)


def _reduce(x, pattern, reduction, **kwargs):
    if reduce is None:
        raise ImportError("mil with TransMIL/transformer needs einops: pip install einops")
    return reduce(x, pattern, reduction, **kwargs)


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = _rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))
    I = torch.eye(x.shape[-1], device=device)
    I = _rearrange(I, "i j -> () i j")
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
        n_token=1,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.n_token = n_token
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads, (residual_conv_kernel, 1), padding=(padding, 0), groups=heads, bias=False
            )

    def forward(self, x, mask=None, return_attn=False):
        b, original_n, _ = x.shape
        n = original_n
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)
            n = n + padding
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: _rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if exists(mask):
            mask = _rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
        q = q * self.scale
        l = ceil(n / m)
        q_landmarks = _reduce(q, "... (n l) d -> ... n d", "sum", l=l)
        k_landmarks = _reduce(k, "... (n l) d -> ... n d", "sum", l=l)
        divisor = l
        if exists(mask):
            mask_landmarks_sum = _reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        q_landmarks /= divisor
        k_landmarks /= divisor
        attn1 = torch.einsum("... i d, ... j d -> ... i j", q, k_landmarks)
        attn2 = torch.einsum("... i d, ... j d -> ... i j", q_landmarks, k_landmarks)
        attn3 = torch.einsum("... i d, ... j d -> ... i j", q_landmarks, k)
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            attn1 = attn1.masked_fill(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn2 = attn2.masked_fill(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn3 = attn3.masked_fill(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)
        if self.residual:
            out = out + self.res_conv(v)
        out = _rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -original_n:]
        if return_attn:
            attn1 = attn1[:, :, : self.n_token] @ attn2
            attn1 = attn1 @ attn3
            return out, attn1.mean(1)
        return out
