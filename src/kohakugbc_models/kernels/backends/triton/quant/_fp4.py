"""Triton NVFP4 producers (``kernels.fp8``'s ``nvfp4``): packed E2M1 codes and e4m3 block scales
in the 128 x 4 blocked layout, the tensor scale ``gs = 448 * 6 / G[0]``.

``PASS`` 1 folds the tile's amax into ``G[0]`` (atomic max); ``PASS`` 2 quantizes with ``G[0]``
(the amax pass's, or a calibrated constant), and ``INV`` writes ``1 / gs`` to ``G[1]``. Per 16
values ``sf = e4m3(min(gs * amax / 6, 448))``, codes ``e2m1(v * gs / sf)`` (``PTX``: the
hardware's ``cvt.rn.satfinite.e2m1x2``, else comparisons; the same codes). Grids cover the rows
rounded up to 128: the padding rows write zero scales.

- ``quant`` / ``act_quant``: tiles of ``BM`` rows by ``BK`` columns.
- ``add_ln_quant``: ``BM`` whole rows. ``PASS`` 1 also stores the stream ``s = x + r`` and each
  row's mean and rstd, so ``PASS`` 2 reads ``s`` alone; ``PASS`` 0 does all in one (a constant
  ``G[0]``).
"""

import triton
import triton.language as tl


@triton.jit
def _e2m1(v):
    a = tl.abs(v)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return code | ((v < 0).to(tl.int32) << 3)


@triton.jit
def _e2m1x2_ptx(lo, hi):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gs(G):
    return 2688.0 / tl.maximum(tl.load(G), 1e-12)


@triton.jit
def _scale_offsets(rows, kb, KBP):
    return (
        ((rows // 128) * (KBP // 4))[:, None] * 512
        + ((kb // 4) * 512)[None, :]
        + ((rows % 32) * 16 + ((rows % 128) // 32) * 4)[:, None]
        + (kb % 4)[None, :]
    )


@triton.jit
def _store(y, rows, k0, M, C, Q, S, gs, BM: tl.constexpr, BK: tl.constexpr, PTX: tl.constexpr):
    """Quantize ``y`` [BM, BK] fp32 (rows ``rows``, columns from ``k0``); codes for rows < M,
    scales for every row given (the padded tile included) and the columns' tail to a multiple
    of 4 blocks zeroed by the chunk that reaches it."""
    nb: tl.constexpr = BK // 16
    KB = C // 16
    KBP = (KB + 3) // 4 * 4
    yb = tl.reshape(y, (BM, nb, 16))
    amax = tl.max(tl.abs(yb), axis=2)
    sf = tl.minimum(gs * amax / 6.0, 448.0).to(tl.float8e4nv)
    sff = sf.to(tl.float32)
    inv = tl.where(sff > 0.0, gs / sff, 0.0)
    lo, hi = tl.split(tl.reshape(yb * inv[:, :, None], (BM, nb, 8, 2)))
    if PTX:  # noqa: SIM108 - a constexpr branch: Triton compiles only the taken side
        packed = _e2m1x2_ptx(lo, hi)
    else:
        packed = _e2m1(lo) | (_e2m1(hi) << 4)
    packed = tl.reshape(packed, (BM, BK // 2)).to(tl.uint8)
    pcols = k0 // 2 + tl.arange(0, BK // 2)
    tl.store(
        Q + rows[:, None] * (C // 2) + pcols[None, :],
        packed,
        mask=(rows < M)[:, None] & (pcols < C // 2)[None, :],
    )
    kb = k0 // 16 + tl.arange(0, nb)
    tl.store(
        S + _scale_offsets(rows, kb, KBP), sf.to(tl.uint8, bitcast=True), mask=(kb < KBP)[None, :]
    )


@triton.jit
def _emit(y, rows, k0, M, C, Q, S, G, PASS: tl.constexpr, INV: tl.constexpr, BM, BK, PTX):
    if PASS == 1:
        tl.atomic_max(G, tl.max(tl.max(tl.abs(y), axis=1), axis=0))
    else:
        gs = _gs(G)
        if INV:  # noqa: SIM102 - a constexpr branch around a runtime one
            if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
                tl.store(G + 1, 1.0 / gs)
        _store(y, rows, k0, M, C, Q, S, gs, BM, BK, PTX)


@triton.jit
def quant(
    X,
    Q,
    S,
    G,
    M,
    C,
    sx,
    ACT: tl.constexpr,
    PASS: tl.constexpr,
    INV: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PTX: tl.constexpr,
):
    """Grid (row tiles over the rows rounded up to 128, column chunks); ``ACT`` 0: the values
    as they are, 1: GELU (erf), 2: ReLU."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    k0 = tl.program_id(1) * BK
    cols = k0 + tl.arange(0, BK)
    mask = (rows < M)[:, None] & (cols < C)[None, :]
    y = tl.load(X + rows[:, None] * sx + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    if ACT == 1:
        y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    elif ACT == 2:
        y = tl.maximum(y, 0.0)
    _emit(y, rows, k0, M, C, Q, S, G, PASS, INV, BM, BK, PTX)


@triton.jit
def add_ln_quant(
    X,
    R,
    W,
    B,
    SOUT,
    STATS,
    Q,
    S,
    G,
    M,
    C,
    eps,
    PASS: tl.constexpr,
    INV: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    PTX: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BC)
    in_m = rows < M
    in_c = cols < C
    mask = in_m[:, None] & in_c[None, :]
    offs = rows[:, None] * C + cols[None, :]
    if PASS == 2:
        sf = tl.load(SOUT + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(STATS + rows * 2, mask=in_m, other=0.0)
        rstd = tl.load(STATS + rows * 2 + 1, mask=in_m, other=0.0)
    else:
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + offs, mask=mask, other=0.0).to(tl.float32)
        s = (x + r).to(SOUT.dtype.element_ty)
        tl.store(SOUT + offs, s, mask=mask)
        sf = s.to(tl.float32)
        mean = tl.sum(sf, axis=1) / C
        d = tl.where(in_c[None, :], sf - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
        if PASS == 1:
            tl.store(STATS + rows * 2, mean, mask=in_m)
            tl.store(STATS + rows * 2 + 1, rstd, mask=in_m)
    w = tl.load(W + cols, mask=in_c, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=in_c, other=0.0).to(tl.float32)
    d = tl.where(in_c[None, :], sf - mean[:, None], 0.0)
    y = tl.where(mask, d * rstd[:, None] * w[None, :] + b[None, :], 0.0)
    if PASS == 1:
        tl.atomic_max(G, tl.max(tl.max(tl.abs(y), axis=1), axis=0))
    else:
        gs = _gs(G)
        if INV:  # noqa: SIM102 - a constexpr branch around a runtime one
            if tl.program_id(0) == 0:
                tl.store(G + 1, 1.0 / gs)
        _store(y, rows, 0, M, C, Q, S, gs, BM, BC, PTX)
