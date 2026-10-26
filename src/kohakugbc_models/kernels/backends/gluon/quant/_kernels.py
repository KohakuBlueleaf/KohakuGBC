"""The ``mx`` producers (Triton): ``[BM, BC]`` tiles quantized from fp32 (one e8m0 scale per 32
along a row, ``2^ceil(log2(amax / 448))``, as ``fp8.quantize``), the scales stored at their
place in cuBLAS's blocked layout (``fp8.to_blocked``: 128-row by 4-scale blocks of 512 bytes;
in a block, row ``r`` and scale ``g`` at ``(r % 32) * 16 + (r % 128 // 32) * 4 + g % 4``).

- ``quant``: ``x`` as is; ``act_quant``: ``ACT`` 1 GELU / 2 ReLU of ``x``, the GELU's erf by
  Abramowitz-Stegun 7.1.26 with the hardware reciprocal and exp2 (absolute error under 5e-7, far
  below an e4m3 step; libdevice's erf made the producer ALU-bound at 2.7x its memory time);
- ``add_ln_quant``: ``s = x + r`` rounded to the stream dtype and stored, the layer norm of the
  rounded ``s`` in fp32 (whole rows, ``BC >= C``), quantized.
"""

import triton
import triton.language as tl


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def gelu(y):
    """``y * (1 + erf(y / sqrt 2)) / 2`` as ``y / 2 + |y| (1 - e) / 2``, ``e = 1 - erf(|y| /
    sqrt 2)`` by Abramowitz-Stegun 7.1.26 with the hardware reciprocal and exp2 (absolute error
    under 5e-7 on [-8, 8])."""
    z = tl.abs(y) * 0.7071067811865476
    t = _rcp(1.0 + 0.3275911 * z)
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    e = poly * _ex2(-z * z * 1.4426950408889634)
    return 0.5 * y + 0.5 * tl.abs(y) * (1.0 - e)


@triton.jit
def _store_mx(y, Q, S, rows, cols, M, C, NC, BM: tl.constexpr, BC: tl.constexpr):
    """``y`` ``[BM, BC]`` fp32 (zero outside ``C``) at ``rows`` / ``cols`` (``cols`` from a
    multiple of 32)."""
    blocks = tl.reshape(y, (BM, BC // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(blocks), axis=2), 1.1754943508222875e-38)
    exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    q = blocks * tl.exp2(-exp)[:, :, None]  # a power of two: the same as the division
    q = tl.reshape(tl.minimum(tl.maximum(q, -448.0), 448.0), (BM, BC))
    keep = (rows < M)[:, None] & (cols < C)[None, :]
    tl.store(
        Q + rows.to(tl.int64)[:, None] * C + cols[None, :], q.to(Q.dtype.element_ty), mask=keep
    )
    g = tl.program_id(1) * (BC // 32) + tl.arange(0, BC // 32)
    off = ((rows // 128) * NC)[:, None] * 512 + ((g // 4) * 512 + g % 4)[None, :]
    off += ((rows % 32) * 16 + (rows % 128 // 32) * 4)[:, None]
    gkeep = (rows < M)[:, None] & (g < C // 32)[None, :]
    tl.store(S + off, (exp + 127.0).to(tl.uint8), mask=gkeep)


@triton.jit
def quant(X, Q, S, M, C, sx, NC, ACT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    keep = (rows < M)[:, None] & (cols < C)[None, :]
    y = tl.load(X + rows.to(tl.int64)[:, None] * sx + cols[None, :], mask=keep, other=0.0)
    y = y.to(tl.float32)
    if ACT == 1:
        y = gelu(y)
    elif ACT == 2:
        y = tl.maximum(y, 0.0)
    _store_mx(y, Q, S, rows, cols, M, C, NC, BM, BC)


@triton.jit
def add_ln_quant(X, R, W, B, SOUT, Q, S, M, C, NC, eps, BM: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BC)
    in_c = cols < C
    keep = (rows < M)[:, None] & in_c[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    x = tl.load(X + off, mask=keep, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=keep, other=0.0).to(tl.float32)
    s = (x + r).to(SOUT.dtype.element_ty)
    tl.store(SOUT + off, s, mask=keep)
    sf = s.to(tl.float32)
    mean = tl.sum(sf, axis=1) / C
    diff = tl.where(keep, sf - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(diff * diff, axis=1) / C + eps)
    w = tl.load(W + cols, mask=in_c, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=in_c, other=0.0).to(tl.float32)
    y = tl.where(keep, diff * rstd[:, None] * w[None, :] + b[None, :], 0.0)
    _store_mx(y, Q, S, rows, cols, M, C, NC, BM, BC)
