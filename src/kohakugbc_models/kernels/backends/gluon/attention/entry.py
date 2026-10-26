"""The attention ops on sm10x.

- ``vit_attention``: unrotated q / k (strided views of the fp8 q / k / v GEMM's output) rotated
  once into contiguous buffers (``rope_copy``); then, global layers (window = grid), KohakuFA's
  tcgen05 forward over the grid as one sequence, or the Triton entry, whichever the tuner
  measured faster; windowed layers the Triton entry.
- ``attention`` without a key mask (the DETR encoder's self-attention, d 32): KohakuFA or the
  Triton entry, tuned per problem; with a mask, the Triton entry.

KohakuFA (``kohakufa``, optional) reads its tuned kernel table from
``<kernel cache>/kohakufa/<card>.json`` when present (``tools``: its ``sm100.tune``), else its
planner's pick.
"""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.gluon.attention import _kernels
from kohakugbc_models.kernels.backends.gluon.gemm.entry import choose
from kohakugbc_models.kernels.backends.triton.attention import entry as triton_attention
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket
from kohakugbc_models.kernels.hardware import cache_dir, card_key

try:
    import kohakufa as fa_package
    from kohakufa.sm100 import tune as kohakufa_tune
except ImportError:  # KohakuFA not installed: the Triton entry alone
    fa_package = kohakufa_tune = None

ROWS = 32
_FA: list = []


def fa_attention(device):
    """KohakuFA's ``attention`` with this card's tuned table loaded, or None without it."""
    if fa_package is None:
        return None
    if not _FA:
        table = cache_dir() / "kohakufa" / f"{card_key(str(torch.device(device)))}.json"
        if table.exists():
            kohakufa_tune.load(str(table))
        _FA.append(fa_package.attention)
    return _FA[0]


def fa_usable(q, k, v) -> bool:
    return q.shape[-1] % 16 == 0 and all(
        t.stride(-1) == 1 and all((s * t.element_size()) % 16 == 0 for s in t.stride()[:-1])
        for t in (q, k, v)
    )


def dense(op: str, key: str, q, k, v, fallback):
    """``[B, H, L, D]`` attention without a mask: KohakuFA or ``fallback``, tuned."""
    fa = fa_attention(q.device)
    if fa is None or not fa_usable(q, k, v):
        return fallback()

    def run_path(name):
        return (lambda: fa(q, k, v)) if name == "kohakufa" else fallback

    config = choose(op, key, q.device, None, run_path, [{"path": "kohakufa"}, {"path": "triton"}])
    return run_path(config["path"])()


def attention(q, k, v, mask=None) -> torch.Tensor:
    if mask is not None:
        return triton_attention.attention(q, k, v, mask)
    batch, heads, lq, dim = q.shape
    key = f"{q.dtype}-h{heads}-d{dim}-q{bucket(lq)}-k{bucket(k.shape[2])}-b{bucket(batch)}"
    return dense("attention", key, q, k, v, lambda: triton_attention.attention(q, k, v))


def supports(query) -> str | None:
    return triton_attention.supports(query)


def rotated_attention(q, k, v, cos, sin, window: int) -> torch.Tensor:
    """Rotated grid operands ``[B, S, S, H, D]``: a global layer over KohakuFA or the Triton
    entry (tuned), a windowed one through the Triton entry."""
    batch, size, _, heads, dim = q.shape

    def fallback():
        return triton_attention.vit_attention(q, k, v, cos, sin, window, rotated=True)

    if window != size:
        return fallback()
    seq = [t.reshape(batch, size * size, heads, dim).transpose(1, 2) for t in (q, k, v)]
    key = f"{q.dtype}-h{heads}-d{dim}-s{size}-b{bucket(batch)}"
    out = dense("vit_global", key, *seq, fallback)
    return out if out.dim() == 5 else out.transpose(1, 2).reshape(q.shape)


def vit_attention(q, k, v, cos, sin, window: int, rotated: bool = False) -> torch.Tensor:
    if rotated:
        return rotated_attention(q, k, v, cos, sin, window)
    if q.stride() != k.stride() or q.stride(-1) != 1:
        return triton_attention.vit_attention(q, k, v, cos, sin, window, rotated)
    batch, size, _, heads, dim = q.shape
    length = window * window
    cos2 = cos.reshape(length, dim).contiguous()
    sin2 = sin.reshape(length, dim).contiguous()
    qa = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ka = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    rows = batch * size * size * heads
    _kernels.rope_copy[(triton.cdiv(rows, ROWS),)](
        q,
        k,
        qa,
        ka,
        cos2,
        sin2,
        rows,
        *q.stride()[:4],
        size,
        WIN=window,
        HEADS=heads,
        D=dim,
        BR=ROWS,
        num_warps=4,
    )
    return rotated_attention(qa, ka, v, cos, sin, window)


registry.register("vit_attention", "gluon", vit_attention, supports)
registry.register("attention", "gluon", attention, supports)
