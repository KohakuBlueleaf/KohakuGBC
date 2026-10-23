"""Activation producers that emit fp8, eager (``kohakugbc_models.kernels.fp8`` recipes): each
returns the e4m3 values ``[M, K]`` and their scales (``row``: fp32 ``[M, 1]``; ``mx``: e8m0 bytes
``[M, K / 32]``), quantized from the fp32 values (no 16-bit round trip).

- ``quant(x, recipe)``: rows of ``x`` ``[..., K]``;
- ``add_ln_quant(x, r, w, b, eps, recipe)``: ``s = x + r`` rounded to the stream dtype, its
  layer norm in fp32, quantized; returns ``(s, q, scales)``;
- ``act_quant(h, act, recipe)``: ``act`` (``gelu``, erf; ``relu``) of ``h`` ``[..., K]`` in
  fp32, quantized.
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.fp8 import quantize


def quant(x: torch.Tensor, recipe: str) -> tuple:
    return quantize(x.reshape(-1, x.shape[-1]), recipe)


def add_ln_quant(x, r, w, b, eps: float, recipe: str) -> tuple:
    s = x + r
    y = F.layer_norm(s.float(), (s.shape[-1],), w.float(), b.float(), eps)
    return (s, *quantize(y.reshape(-1, y.shape[-1]), recipe))


ACTS = {"gelu": F.gelu, "relu": F.relu}


def act_quant(h: torch.Tensor, act: str, recipe: str) -> tuple:
    return quantize(ACTS[act](h.float()).reshape(-1, h.shape[-1]), recipe)


registry.register("quant", "eager", quant)
registry.register("add_ln_quant", "eager", add_ln_quant)
registry.register("act_quant", "eager", act_quant)
