"""Triton block kernels: 16-bit operands, fp32 accumulators, each value the eager path rounds
rounded at the same point.

``gemm``: ``X [M, K] @ W[N, K].T + bias`` rounded to the output dtype (as ``F.linear``), then in
fp32: ``ACT`` 0 none / 1 GELU (erf) / 2 ReLU; ``ROPE``: the ViT's fused q / k / v projection
(``N = 3C``), columns ``< 2C`` rotated pairwise (accumulator pairs split, rotated, joined) by
the row's in-window position (row ``m`` is grid cell ``m % SIZE^2``), stored as ``[3, M, C]``.
Tiles are walked in ``GROUP``-row bands so a band's W tiles stay in L2.

``add_layer_norm``: ``RB`` rows per program; ``s = x + r`` rounded to the stream dtype and
stored, then the layer norm of the rounded ``s`` in fp32 (two-pass, row in registers).
"""

import triton
import triton.language as tl


@triton.jit
def _tile(M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    band = GROUP * grid_n
    first = (pid // band) * GROUP
    rows = tl.minimum(grid_m - first, GROUP)
    return first + (pid % band) % rows, (pid % band) // rows


@triton.jit
def gemm(
    X,
    W,
    BIAS,
    OUT,
    COS,
    SIN,
    M,
    N,
    K,
    sxm,
    swn,
    som,
    C,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    ROPE: tl.constexpr,
    SIZE: tl.constexpr,
    WIN: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_m, pid_n = _tile(M, N, BM, BN, GROUP)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)  # M * N passes 2^31 at large batches
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    in_m = offs_m < M
    in_n = offs_n < N
    x_ptr = X + offs_m[:, None] * sxm + offs_k[None, :]
    w_ptr = W + offs_n[None, :] * swn + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        in_k = offs_k < K - k * BK
        x = tl.load(x_ptr, mask=in_m[:, None] & in_k[None, :], other=0.0)
        w = tl.load(w_ptr, mask=in_n[None, :] & in_k[:, None], other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptr += BK
        w_ptr += BK
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=in_n, other=0.0).to(tl.float32)[None, :]
    y = acc.to(OUT.dtype.element_ty).to(tl.float32)
    if ACT == 1:
        y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    elif ACT == 2:
        y = tl.maximum(y, 0.0)
    if ROPE:
        cell = offs_m % (SIZE * SIZE)
        t = ((cell // SIZE) % WIN) * WIN + (cell % SIZE) % WIN
        even = pid_n * BN + 2 * tl.arange(0, BN // 2)
        table = t[:, None] * D + (even % D)[None, :]
        keep = in_m[:, None]
        ce = tl.load(COS + table, mask=keep, other=0.0)
        se = tl.load(SIN + table, mask=keep, other=0.0)
        co = tl.load(COS + table + 1, mask=keep, other=0.0)
        so = tl.load(SIN + table + 1, mask=keep, other=0.0)
        ye, yo = tl.split(tl.reshape(y, (BM, BN // 2, 2)))
        rot = tl.reshape(tl.join(ye * ce - yo * se, yo * co + ye * so), (BM, BN))
        y = tl.where((offs_n < 2 * C)[None, :], rot, y)
        mat = (offs_n // C)[None, :]
        out_ptr = OUT + mat * (M * C) + offs_m[:, None] * C + (offs_n % C)[None, :]
    else:
        out_ptr = OUT + offs_m[:, None] * som + offs_n[None, :]
    tl.store(out_ptr, y.to(OUT.dtype.element_ty), mask=in_m[:, None] & in_n[None, :])


@triton.jit
def add_layer_norm(
    X,
    R,
    W,
    B,
    S,
    Y,
    rows,
    eps,
    C: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    BC: tl.constexpr,
    RB: tl.constexpr,
):
    idx = tl.program_id(0) * RB + tl.arange(0, RB)
    col = tl.arange(0, BC)
    in_c = col < C
    keep = (idx < rows)[:, None] & in_c[None, :]
    off = idx[:, None] * C + col[None, :]
    x = tl.load(X + off, mask=keep, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=keep, other=0.0).to(tl.float32)
    s = (x + r).to(S.dtype.element_ty)
    tl.store(S + off, s, mask=keep)
    sf = s.to(tl.float32)
    mean = tl.sum(sf, axis=1) / C
    diff = tl.where(keep, sf - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(diff * diff, axis=1) / C + eps)
    y = diff * rstd[:, None]
    if HAS_W:
        y = y * tl.load(W + col, mask=in_c, other=0.0).to(tl.float32)[None, :]
    if HAS_B:
        y = y + tl.load(B + col, mask=in_c, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=keep)
