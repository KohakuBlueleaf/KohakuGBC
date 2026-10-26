"""Triton fp8 producers (``row`` / ``mx``): one row per program, the row in registers (``BC`` >=
its width), the values quantized from fp32 (``ROW``: one scale per row; else MX, one e8m0 scale
per 32). The NVFP4 producers are ``_fp4``'s.
"""

import triton
import triton.language as tl


@triton.jit
def _store_quant(y, Q, S, row, cols, in_c, C, ROW: tl.constexpr, BC: tl.constexpr):
    if ROW:
        amax = tl.max(tl.where(in_c, tl.abs(y), 0.0), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        q = tl.minimum(tl.maximum(y / scale, -448.0), 448.0)
        tl.store(Q + row * C + cols, q.to(Q.dtype.element_ty), mask=in_c)
        tl.store(S + row, scale)
    else:
        blocks = tl.reshape(tl.where(in_c, y, 0.0), (BC // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(blocks), axis=1), 1.1754943508222875e-38)
        exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
        q = blocks / tl.exp2(exp)[:, None]
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(Q + row * C + cols, tl.reshape(q, (BC,)).to(Q.dtype.element_ty), mask=in_c)
        groups = tl.arange(0, BC // 32)
        tl.store(S + row * (C // 32) + groups, (exp + 127.0).to(tl.uint8), mask=groups < C // 32)


@triton.jit
def quant(X, Q, S, sx, C, ROW: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BC)
    in_c = cols < C
    y = tl.load(X + row * sx + cols, mask=in_c, other=0.0).to(tl.float32)
    _store_quant(y, Q, S, row, cols, in_c, C, ROW, BC)


@triton.jit
def act_quant(X, Q, S, sx, C, ACT: tl.constexpr, ROW: tl.constexpr, BC: tl.constexpr):
    """``ACT`` 1: GELU (erf); 2: ReLU."""
    row = tl.program_id(0)
    cols = tl.arange(0, BC)
    in_c = cols < C
    h = tl.load(X + row * sx + cols, mask=in_c, other=0.0).to(tl.float32)
    y = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476)) if ACT == 1 else tl.maximum(h, 0.0)
    _store_quant(y, Q, S, row, cols, in_c, C, ROW, BC)


@triton.jit
def add_ln_quant(X, R, W, B, SOUT, Q, S, C, eps, ROW: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BC)
    in_c = cols < C
    x = tl.load(X + row * C + cols, mask=in_c, other=0.0).to(tl.float32)
    r = tl.load(R + row * C + cols, mask=in_c, other=0.0).to(tl.float32)
    s = (x + r).to(SOUT.dtype.element_ty)
    tl.store(SOUT + row * C + cols, s, mask=in_c)
    sf = s.to(tl.float32)
    mean = tl.sum(sf, axis=0) / C
    diff = tl.where(in_c, sf - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(diff * diff, axis=0) / C + eps)
    w = tl.load(W + cols, mask=in_c, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=in_c, other=0.0).to(tl.float32)
    _store_quant(diff * rstd * w + b, Q, S, row, cols, in_c, C, ROW, BC)
