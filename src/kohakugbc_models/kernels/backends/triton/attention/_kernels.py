"""Triton attention forwards (inference): 16-bit operands, fp32 scores and accumulators, the
probabilities rounded once to the operands' dtype before P . V (as flash attention does).

``vit_fwd``: SAM 3's ViT attention on grid-layout operands ``[B, S, S, H, D]``; program
``(m, b * nW * H)`` takes ``BM`` tokens of one window of one head; token ``t`` of window
``(wy, wx)`` is grid cell ``(wy * WIN + t // WIN, wx * WIN + t % WIN)``. With ``ROPE``, q and k
tiles get the pairwise 2D rotary embedding in fp32 (``cos`` / ``sin`` ``[WIN^2, D]`` by
in-window position), rounded back to the operands' dtype before the dot, as the eager path's
cast does; every query block then re-rotates every key tile. ``vit_rope`` rotates q and k once
instead (one pass, grid layout), for ``vit_fwd`` with ``ROPE=False``.

``attn_fwd``: ``[B, H, L, D]`` operands (any strides, unit stride on D); ``HAS_MASK``: a bool key
mask ``[B, S]`` (False keys excluded); ``HAS_RPB``: a separable bias from tables ``[B, H, L, GH]``
/ ``[B, H, L, GW]`` (any strides: the decoder's come out of a padded permute), key ``n`` reading
``y[n // GW] + x[n % GW]`` (summed in fp32).
"""

import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _rope(x, xs, cos, sin, d):
    """``x * cos + rotate_pairwise(x) * sin`` in fp32: ``xs`` is ``x`` loaded at column
    ``d ^ 1``; even columns take ``-x[d + 1]``, odd ones ``x[d - 1]``."""
    sign = tl.where(d % 2 == 0, -1.0, 1.0)
    return x.to(tl.float32) * cos + xs.to(tl.float32) * sign[None, :] * sin


@triton.jit
def vit_rope(
    Q,
    K,
    QO,
    KO,
    COS,
    SIN,
    rows,
    sr,
    sc,
    sh,
    size,
    WIN: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    """q and k rotated once, in grid layout: ``BR`` (batch, row, col, head) rows per program;
    a row's table entry is its in-window position ``(r % WIN) * WIN + c % WIN``. Contiguous
    ``[B, S, S, H, D]`` operands and outputs (``sr`` / ``sc`` / ``sh`` their strides)."""
    pid = tl.program_id(0)
    d = tl.arange(0, D)
    idx = pid * BR + tl.arange(0, BR)
    keep = idx < rows
    h = idx % HEADS
    cell = idx // HEADS
    c = cell % size
    r = (cell // size) % size
    b = cell // (size * size)
    t = (r % WIN) * WIN + c % WIN
    base = b * (size * sr) + r * sr + c * sc + h * sh
    cos = tl.load(COS + t[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
    sin = tl.load(SIN + t[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
    _rope_rows(Q, QO, base, cos, sin, keep, d)
    _rope_rows(K, KO, base, cos, sin, keep, d)


@triton.jit
def _rope_rows(X, XO, base, cos, sin, keep, d):
    x = tl.load(X + base[:, None] + d[None, :], mask=keep[:, None], other=0.0)
    xs = tl.load(X + base[:, None] + (d ^ 1)[None, :], mask=keep[:, None], other=0.0)
    tl.store(
        XO + base[:, None] + d[None, :],
        _rope(x, xs, cos, sin, d).to(XO.dtype.element_ty),
        mask=keep[:, None],
    )


@triton.jit
def vit_fwd(
    Q,
    K,
    V,
    OUT,
    COS,
    SIN,
    sqb,
    sqr,
    sqc,
    sqh,
    skb,
    skr,
    skc,
    skh,
    svb,
    svr,
    svc,
    svh,
    sob,
    sor,
    soc,
    soh,
    scale,
    NWX: tl.constexpr,
    WIN: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROPE: tl.constexpr = True,
):
    L: tl.constexpr = WIN * WIN
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    h = pid % HEADS
    w = (pid // HEADS) % (NWX * NWX)
    b = pid // (HEADS * NWX * NWX)
    wy = w // NWX
    wx = w % NWX
    d = tl.arange(0, D)

    offs_m = pid_m * BM + tl.arange(0, BM)
    in_m = offs_m < L
    rm = wy * WIN + offs_m // WIN
    cm = wx * WIN + offs_m % WIN
    q_base = Q + b * sqb + rm[:, None] * sqr + cm[:, None] * sqc + h * sqh
    q = tl.load(q_base + d[None, :], mask=in_m[:, None], other=0.0)
    if ROPE:
        qs = tl.load(q_base + (d ^ 1)[None, :], mask=in_m[:, None], other=0.0)
        cos = tl.load(COS + offs_m[:, None] * D + d[None, :], mask=in_m[:, None], other=0.0)
        sin = tl.load(SIN + offs_m[:, None] * D + d[None, :], mask=in_m[:, None], other=0.0)
        q = _rope(q, qs, cos, sin, d).to(Q.dtype.element_ty)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for start in range(0, L, BN):
        offs_n = start + tl.arange(0, BN)
        in_n = offs_n < L
        rn = wy * WIN + offs_n // WIN
        cn = wx * WIN + offs_n % WIN
        k_base = K + b * skb + rn[:, None] * skr + cn[:, None] * skc + h * skh
        k = tl.load(k_base + d[None, :], mask=in_n[:, None], other=0.0)
        if ROPE:
            ks = tl.load(k_base + (d ^ 1)[None, :], mask=in_n[:, None], other=0.0)
            kcos = tl.load(COS + offs_n[:, None] * D + d[None, :], mask=in_n[:, None], other=0.0)
            ksin = tl.load(SIN + offs_n[:, None] * D + d[None, :], mask=in_n[:, None], other=0.0)
            k = _rope(k, ks, kcos, ksin, d).to(K.dtype.element_ty)
        v_base = V + b * svb + rn[:, None] * svr + cn[:, None] * svc + h * svh
        v = tl.load(v_base + d[None, :], mask=in_n[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(in_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_new

    out = acc / l_i[:, None]
    o_base = OUT + b * sob + rm[:, None] * sor + cm[:, None] * soc + h * soh
    tl.store(o_base + d[None, :], out.to(OUT.dtype.element_ty), mask=in_m[:, None])


@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    OUT,
    MASK,
    Y,
    X,
    sqb,
    sqh,
    sql,
    skb,
    skh,
    skl,
    svb,
    svh,
    svl,
    sob,
    soh,
    sol,
    smb,
    syb,
    syh,
    syl,
    syg,
    sxb,
    sxh,
    sxl,
    sxg,
    LQ,
    LK,
    scale,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    GW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_RPB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    h = pid % HEADS
    b = pid // HEADS
    d = tl.arange(0, D)
    offs_m = pid_m * BM + tl.arange(0, BM)
    in_m = offs_m < LQ
    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sql + d[None, :], mask=in_m[:, None], other=0.0
    )

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for start in range(0, LK, BN):
        offs_n = start + tl.arange(0, BN)
        in_n = offs_n < LK
        k = tl.load(
            K + b * skb + h * skh + offs_n[:, None] * skl + d[None, :],
            mask=in_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + b * svb + h * svh + offs_n[:, None] * svl + d[None, :],
            mask=in_n[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = in_n
        if HAS_MASK:
            keep = keep & (tl.load(MASK + b * smb + offs_n, mask=in_n, other=0) != 0)
        if HAS_RPB:
            y = tl.load(
                Y + b * syb + h * syh + offs_m[:, None] * syl + (offs_n // GW)[None, :] * syg,
                mask=in_m[:, None] & in_n[None, :],
                other=0.0,
            )
            x = tl.load(
                X + b * sxb + h * sxh + offs_m[:, None] * sxl + (offs_n % GW)[None, :] * sxg,
                mask=in_m[:, None] & in_n[None, :],
                other=0.0,
            )
            s += (y.to(tl.float32) + x.to(tl.float32)) * 1.4426950408889634
        s = tl.where(keep[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        OUT + b * sob + h * soh + offs_m[:, None] * sol + d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=in_m[:, None],
    )
