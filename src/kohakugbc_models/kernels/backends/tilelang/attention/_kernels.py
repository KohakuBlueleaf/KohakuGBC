"""TileLang attention forwards (Volta / Turing MMA; fp16 operands, fp32 scores and
accumulators, probabilities rounded once to fp16 before P . V), the same semantics as the
Triton kernels.

``vit_fwd``: grid-layout ``[B, S, S, H, D]`` operands, windows of ``WIN x WIN`` tokens (token
``t`` of window ``(wy, wx)`` at grid cell ``(wy * WIN + t // WIN, wx * WIN + t % WIN)``), the
pairwise rotary embedding applied to q and k in fp32 between the gather and the tile.

``attn_fwd``: operands in their physical ``[B, L, H, D]`` layout; ``has_mask``: a key mask
``[B, S]`` (int8, 0 excluded); ``has_rpb``: tables ``[B, H, LQ, GH]`` / ``[B, H, LQ, GW]``, key
``n`` reading ``y[n // GW] + x[n % GW]`` (summed in fp32).
"""

import tilelang
import tilelang.language as T

ACC = "float32"
LOG2E = 1.4426950408889634


def compile_kernel(prim, arch: str):
    return tilelang.compile(
        prim,
        target={"kind": "cuda", "arch": arch},
        pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True},
    )


def vit_fwd(
    nwx: int,
    win: int,
    heads: int,
    d: int,
    dtype: str,
    bm: int,
    bn: int,
    threads: int,
    rope: bool = True,
):
    b = T.dynamic("b")
    size = nwx * win
    length = win * win

    @T.prim_func
    def kernel(
        Q: T.Tensor((b, size, size, heads, d), dtype),
        K: T.Tensor((b, size, size, heads, d), dtype),
        V: T.Tensor((b, size, size, heads, d), dtype),
        COS: T.Tensor((length, d), ACC),
        SIN: T.Tensor((length, d), ACC),
        Out: T.Tensor((b, size, size, heads, d), dtype),
        c: T.float32,
    ):
        with T.Kernel(T.ceildiv(length, bm), b * nwx * nwx * heads, threads=threads) as (
            mb,
            pid,
        ):
            h = pid % heads
            w = (pid // heads) % (nwx * nwx)
            bb = pid // (heads * nwx * nwx)
            wy = w // nwx
            wx = w % nwx
            raw = T.alloc_shared((bm, d), dtype)
            q_s = T.alloc_shared((bm, d), dtype)
            kraw = T.alloc_shared((bn, d), dtype)
            k_s = T.alloc_shared((bn, d), dtype)
            v_s = T.alloc_shared((bn, d), dtype)
            s = T.alloc_fragment((bm, bn), ACC)
            p_s = T.alloc_shared((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), ACC)
            m_i = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            alpha = T.alloc_fragment((bm,), ACC)
            row_sum = T.alloc_fragment((bm,), ACC)
            l_i = T.alloc_fragment((bm,), ACC)

            for i, j in T.Parallel(bm, d):
                t = mb * bm + i
                raw[i, j] = T.if_then_else(
                    t < length, Q[bb, wy * win + t // win, wx * win + t % win, h, j], 0
                )
            for i, j in T.Parallel(bm, d):
                t = mb * bm + i
                if rope:
                    partner = j + 1 - 2 * (j % 2)
                    sign = T.if_then_else(j % 2 == 0, -1.0, 1.0)
                    q_s[i, j] = T.if_then_else(
                        t < length,
                        raw[i, j].astype(ACC) * COS[t, j]
                        + sign * raw[i, partner].astype(ACC) * SIN[t, j],
                        0,
                    ).astype(dtype)
                else:
                    q_s[i, j] = raw[i, j]
            T.fill(acc, 0)
            T.fill(l_i, 0)
            T.fill(m_i, -T.infinity(ACC))
            for kb in T.serial(T.ceildiv(length, bn)):
                for i, j in T.Parallel(bn, d):
                    t = kb * bn + i
                    kraw[i, j] = T.if_then_else(
                        t < length, K[bb, wy * win + t // win, wx * win + t % win, h, j], 0
                    )
                    v_s[i, j] = T.if_then_else(
                        t < length, V[bb, wy * win + t // win, wx * win + t % win, h, j], 0
                    )
                for i, j in T.Parallel(bn, d):
                    t = kb * bn + i
                    if rope:
                        partner = j + 1 - 2 * (j % 2)
                        sign = T.if_then_else(j % 2 == 0, -1.0, 1.0)
                        k_s[i, j] = T.if_then_else(
                            t < length,
                            kraw[i, j].astype(ACC) * COS[t, j]
                            + sign * kraw[i, partner].astype(ACC) * SIN[t, j],
                            0,
                        ).astype(dtype)
                    else:
                        k_s[i, j] = kraw[i, j]
                T.clear(s)
                T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    s[i, j] = T.if_then_else(kb * bn + j < length, s[i, j], -T.infinity(ACC))
                T.copy(m_i, m_prev)
                T.reduce_max(s, m_i, dim=1, clear=False)
                for i in T.Parallel(bm):
                    alpha[i] = T.exp2((m_prev[i] - m_i[i]) * c)
                for i, j in T.Parallel(bm, bn):
                    s[i, j] = T.exp2((s[i, j] - m_i[i]) * c)
                T.reduce_sum(s, row_sum, dim=1)
                for i in T.Parallel(bm):
                    l_i[i] = l_i[i] * alpha[i] + row_sum[i]
                for i, j in T.Parallel(bm, d):
                    acc[i, j] *= alpha[i]
                T.copy(s, p_s)
                T.gemm(p_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, d):
                t = mb * bm + i
                if t < length:
                    Out[bb, wy * win + t // win, wx * win + t % win, h, j] = acc[i, j] / l_i[i]

    return kernel


def attn_fwd(
    heads: int,
    d: int,
    dtype: str,
    has_mask: bool,
    has_rpb: bool,
    gh: int,
    gw: int,
    bm: int,
    bn: int,
    threads: int,
):
    b = T.dynamic("b")
    lq = T.dynamic("lq")
    lk = T.dynamic("lk")

    @T.prim_func
    def kernel(
        Q: T.Tensor((b, lq, heads, d), dtype),
        K: T.Tensor((b, lk, heads, d), dtype),
        V: T.Tensor((b, lk, heads, d), dtype),
        MASK: T.Tensor((b, lk), "int8"),
        Y: T.Tensor((b, heads, lq, gh), dtype),
        X: T.Tensor((b, heads, lq, gw), dtype),
        Out: T.Tensor((b, lq, heads, d), dtype),
        c: T.float32,
    ):
        with T.Kernel(T.ceildiv(lq, bm), b * heads, threads=threads) as (mb, pid):
            h = pid % heads
            bb = pid // heads
            q_s = T.alloc_shared((bm, d), dtype)
            k_s = T.alloc_shared((bn, d), dtype)
            v_s = T.alloc_shared((bn, d), dtype)
            s = T.alloc_fragment((bm, bn), ACC)
            p_s = T.alloc_shared((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), ACC)
            m_i = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            alpha = T.alloc_fragment((bm,), ACC)
            row_sum = T.alloc_fragment((bm,), ACC)
            l_i = T.alloc_fragment((bm,), ACC)
            for i, j in T.Parallel(bm, d):
                q_s[i, j] = T.if_then_else(mb * bm + i < lq, Q[bb, mb * bm + i, h, j], 0)
            T.fill(acc, 0)
            T.fill(l_i, 0)
            T.fill(m_i, -T.infinity(ACC))
            for kb in T.serial(T.ceildiv(lk, bn)):
                for i, j in T.Parallel(bn, d):
                    k_s[i, j] = T.if_then_else(kb * bn + i < lk, K[bb, kb * bn + i, h, j], 0)
                    v_s[i, j] = T.if_then_else(kb * bn + i < lk, V[bb, kb * bn + i, h, j], 0)
                T.clear(s)
                T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    n = kb * bn + j
                    r = mb * bm + i
                    if has_rpb:
                        # the bias in raw-score units: (s * scale + bias) * log2e = (s + bias
                        # * log2e / c) * c
                        s[i, j] = T.if_then_else(
                            n < lk,
                            T.if_then_else(
                                r < lq,
                                s[i, j]
                                + (
                                    Y[bb, h, r, n // gw].astype(ACC)
                                    + X[bb, h, r, n % gw].astype(ACC)
                                )
                                * LOG2E
                                / c,
                                s[i, j],
                            ),
                            -T.infinity(ACC),
                        )
                    elif has_mask:
                        s[i, j] = T.if_then_else(
                            n < lk,
                            T.if_then_else(MASK[bb, n] != 0, s[i, j], -T.infinity(ACC)),
                            -T.infinity(ACC),
                        )
                    else:
                        s[i, j] = T.if_then_else(n < lk, s[i, j], -T.infinity(ACC))
                T.copy(m_i, m_prev)
                T.reduce_max(s, m_i, dim=1, clear=False)
                for i in T.Parallel(bm):
                    alpha[i] = T.exp2((m_prev[i] - m_i[i]) * c)
                for i, j in T.Parallel(bm, bn):
                    s[i, j] = T.exp2((s[i, j] - m_i[i]) * c)
                T.reduce_sum(s, row_sum, dim=1)
                for i in T.Parallel(bm):
                    l_i[i] = l_i[i] * alpha[i] + row_sum[i]
                for i, j in T.Parallel(bm, d):
                    acc[i, j] *= alpha[i]
                T.copy(s, p_s)
                T.gemm(p_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, d):
                if mb * bm + i < lq:
                    Out[bb, mb * bm + i, h, j] = acc[i, j] / l_i[i]

    return kernel
