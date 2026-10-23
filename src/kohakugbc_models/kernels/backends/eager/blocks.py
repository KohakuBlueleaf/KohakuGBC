"""Block ops, eager: what each computes, with the casts of transformers' SAM 3.

- ``linear_act(x, w, b, act)``: ``act(x @ w.T + b)``, ``act`` ``"none"`` / ``"gelu"`` (erf) /
  ``"relu"``; the linear rounds to x's dtype before the activation, as ``F.linear`` then
  ``F.gelu`` do.
- ``qkv_rope(x, w, b, cos, sin, window)``: the ViT's q / k / v projections of grid features
  ``[B, S, S, C]`` with one weight ``[3C, C]`` (rows q, k, v) and its bias; q and k get SAM 3's
  pairwise 2D rotary embedding by in-window position (``cos`` / ``sin`` ``[1, window^2, D]``,
  fp32, cast back), heads of ``D = cos.shape[-1]``; returns q, k, v as ``[B, S, S, H, D]``.
- ``add_layer_norm(x, r, w, b, eps)``: ``s = x + r`` (rounded to x's dtype) and
  ``layer_norm(s)``; returns ``(s, normed)``.
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.eager.attention import rotate_pairwise

ACTS = {"none": lambda t: t, "gelu": F.gelu, "relu": F.relu}


def linear_act(x: torch.Tensor, w: torch.Tensor, b, act: str = "none") -> torch.Tensor:
    return ACTS[act](F.linear(x, w, b))


def window_tables(cos: torch.Tensor, sin: torch.Tensor, size: int, window: int) -> tuple:
    """Per grid cell ``[S, S, 1, D]`` tables: cell ``(r, c)`` takes in-window position
    ``(r % window) * window + c % window``."""
    d = cos.shape[-1]
    idx = torch.arange(size, device=cos.device) % window
    pos = (idx[:, None] * window + idx[None, :]).flatten()
    return (
        cos.reshape(-1, d)[pos].view(size, size, 1, d),
        sin.reshape(-1, d)[pos].view(size, size, 1, d),
    )


def qkv_rope(x: torch.Tensor, w: torch.Tensor, b, cos, sin, window: int) -> tuple:
    batch, size, _, channels = x.shape
    dim = cos.shape[-1]
    shape = (batch, size, size, channels // dim, dim)
    wq, wk, wv = w.chunk(3)
    bq, bk, bv = (None, None, None) if b is None else b.chunk(3)
    q = F.linear(x, wq, bq).view(shape)
    k = F.linear(x, wk, bk).view(shape)
    v = F.linear(x, wv, bv).view(shape)
    c, s = window_tables(cos, sin, size, window)
    q32, k32 = q.float(), k.float()
    q = (q32 * c + rotate_pairwise(q32) * s).type_as(q)
    k = (k32 * c + rotate_pairwise(k32) * s).type_as(k)
    return q, k, v


def add_layer_norm(x: torch.Tensor, r: torch.Tensor, w, b, eps: float) -> tuple:
    s = x + r
    return s, F.layer_norm(s, (s.shape[-1],), w, b, eps)


registry.register("linear_act", "eager", linear_act)
registry.register("qkv_rope", "eager", qkv_rope)
registry.register("add_layer_norm", "eager", add_layer_norm)
