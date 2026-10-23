"""Triton kernels (sm80+, XPU). The probe builds and launches one minimal kernel per device."""

import torch
import triton
import triton.language as tl

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends import load_tree


@triton.jit
def _probe_kernel(x, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x + offs, tl.load(x + offs, mask=offs < n) + 1, mask=offs < n)


def probe(device: str) -> str | None:
    try:
        x = torch.zeros(16, device=device)
        _probe_kernel[(1,)](x, 16, BLOCK=16)
        if x.sum().item() != 16:
            return "probe kernel computed a wrong result"
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return None


registry.register_probe("triton", probe)
FAILED = load_tree(__name__, __path__)
