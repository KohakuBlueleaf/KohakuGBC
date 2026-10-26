"""``rope_copy`` (Triton): q and k of grid layout ``[B, S, S, H, D]`` read at any strides (views
of a fused q / k / v GEMM output), rotated pairwise by in-window position as ``vit_rope`` does,
written to contiguous buffers: one pass where a copy and an in-place rotation take two."""

import triton
import triton.language as tl


@triton.jit
def _rows(X, XO, src, dst, cos, sin, keep, d, BR: tl.constexpr, D: tl.constexpr):
    """One load per element: the pairs split in registers, rotated (``vit_rope``'s
    ``x * cos + rotate_pairwise(x) * sin`` in fp32), joined back."""
    x = tl.load(X + src[:, None] + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
    xe, xo = tl.split(tl.reshape(x, (BR, D // 2, 2)))
    ce, co = tl.split(tl.reshape(cos, (BR, D // 2, 2)))
    se, so = tl.split(tl.reshape(sin, (BR, D // 2, 2)))
    out = tl.reshape(tl.join(xe * ce - xo * se, xo * co + xe * so), (BR, D))
    tl.store(XO + dst[:, None] + d[None, :], out.to(XO.dtype.element_ty), mask=keep[:, None])


@triton.jit
def rope_copy(
    Q,
    K,
    QO,
    KO,
    COS,
    SIN,
    rows,
    sb,
    sr,
    sc,
    sh,
    size,
    WIN: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    """``BR`` (batch, row, col, head) rows per program; q and k share the strides ``sb`` /
    ``sr`` / ``sc`` / ``sh``; outputs contiguous."""
    d = tl.arange(0, D)
    idx = tl.program_id(0) * BR + tl.arange(0, BR)
    keep = idx < rows
    h = idx % HEADS
    cell = idx // HEADS
    c = cell % size
    r = (cell // size) % size
    b = cell // (size * size)
    t = (r % WIN) * WIN + c % WIN
    src = b.to(tl.int64) * sb + r * sr + c * sc + h * sh
    dst = idx.to(tl.int64) * D
    cos = tl.load(COS + t[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
    sin = tl.load(SIN + t[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
    _rows(Q, QO, src, dst, cos, sin, keep, d, BR, D)
    _rows(K, KO, src, dst, cos, sin, keep, d, BR, D)
