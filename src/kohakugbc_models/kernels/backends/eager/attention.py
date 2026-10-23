"""Attention ops, eager: what each computes, in the operation order of transformers' SAM 3.

- ``attention(q, k, v, mask=None)``: ``[B, H, L, D]`` operands (any strides), scale
  ``D ** -0.5``; ``mask`` None or bool ``[B, 1, 1 or L, S]`` (True attends).
- ``vit_attention(q, k, v, cos, sin, window, rotated=False)``: SAM 3's ViT attention on
  grid-layout operands ``[B, S, S, H, D]``: tokens attend within ``window x window`` windows
  (``window == S``: global), the pairwise 2D rotary embedding (``cos`` / ``sin`` ``[1,
  window^2, D]`` fp32, by the token's position in its window) applied to q and k in fp32 and
  cast back unless they come ``rotated`` (``qkv_rope``); the output in the same grid layout.
- ``attention_rpb(q, k, v, bias_y, bias_x)``: attention plus the box relative-position bias of
  SAM 3's decoder given by its separable tables ``[B, H, L, h]`` and ``[B, H, L, w]``: key
  ``(r, c)`` of the ``h x w`` grid gets ``bias_y[..., r] + bias_x[..., c]`` (summed in the
  tables' dtype, as transformers materializes it).
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kohakugbc_models.kernels import registry

_PRE_AMPERE: dict[torch.device, bool] = {}


def pre_ampere(device: torch.device) -> bool:
    if device not in _PRE_AMPERE:
        cuda = device.type == "cuda"
        _PRE_AMPERE[device] = cuda and torch.cuda.get_device_capability(device)[0] < 8
    return _PRE_AMPERE[device]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    if pre_ampere(q.device):
        # torch 2.14's cu130 build picks FlashAttention on Turing and fails ("only supports
        # Ampere GPUs or newer") instead of falling back: its other kernels, named
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    q32, k32 = q.float(), k.float()
    q_out = q32 * cos + rotate_pairwise(q32) * sin
    k_out = k32 * cos + rotate_pairwise(k32) * sin
    return q_out.type_as(q), k_out.type_as(k)


def windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """``[B, S, S, H, D]`` to ``[B * nW, H, window^2, D]`` (raster order inside a window)."""
    batch, size, _, heads, dim = x.shape
    n = size // window
    x = x.view(batch, n, window, n, window, heads, dim).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(batch * n * n, heads, window * window, dim)


def grid(x: torch.Tensor, batch: int, size: int, window: int) -> torch.Tensor:
    """The inverse of ``windows``: ``[B * nW, H, window^2, D]`` to ``[B, S, S, H, D]``."""
    n = size // window
    heads, dim = x.shape[1], x.shape[3]
    x = x.view(batch, n, n, heads, window, window, dim).permute(0, 1, 4, 2, 5, 3, 6)
    return x.reshape(batch, size, size, heads, dim)


def vit_attention(q, k, v, cos, sin, window: int, rotated: bool = False) -> torch.Tensor:
    batch, size = q.shape[0], q.shape[1]
    q, k, v = (windows(t, window) for t in (q, k, v))
    if not rotated:
        q, k = rope(q, k, cos, sin)
    return grid(attention(q, k, v), batch, size, window)


def attention_rpb(q, k, v, bias_y: torch.Tensor, bias_x: torch.Tensor) -> torch.Tensor:
    bias = (bias_y.unsqueeze(-1) + bias_x.unsqueeze(-2)).flatten(-2)
    return attention(q, k, v, bias)


registry.register("attention", "eager", attention)
registry.register("vit_attention", "eager", vit_attention)
registry.register("attention_rpb", "eager", attention_rpb)
