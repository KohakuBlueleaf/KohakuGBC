"""Triton packed variable-length kernels: 16-bit operands, fp32 scores and accumulators.

``varlen_fwd``: program ``(m, h)`` takes ``BM`` consecutive tokens of the packed stream for
query head ``h`` (its key head ``h // GROUP``). Rows of one block may belong to several short
sequences: the key loop runs over the union of the rows' spans (narrowed by the window), and
each score keeps its own row's span and window. Probabilities are rounded once to the operands'
dtype before P . V.

``segment_mean``: program ``(n, c)`` sums the rows of segment ``n`` in fp32, ``BC`` columns.
"""

import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def varlen_fwd(
    Q,
    K,
    V,
    OUT,
    START,
    END,
    sqt,
    sqh,
    skt,
    skh,
    svt,
    svh,
    sot,
    soh,
    total,
    window,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    kh = h // GROUP
    d = tl.arange(0, D)
    rows = pid_m * BM + tl.arange(0, BM)
    in_m = rows < total
    st = tl.load(START + rows, mask=in_m, other=total)
    en = tl.load(END + rows, mask=in_m, other=0)
    lo = tl.min(st, axis=0)
    hi = tl.max(en, axis=0)
    if window > 0:
        lo = tl.maximum(lo, pid_m * BM - window + 1)
        hi = tl.minimum(hi, pid_m * BM + BM + window - 1)
    lo = tl.maximum(lo, 0) // BN * BN
    q = tl.load(Q + rows[:, None] * sqt + h * sqh + d[None, :], mask=in_m[:, None], other=0.0)
    c = scale * LOG2E
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for n in range(lo, hi, BN):
        cols = n + tl.arange(0, BN)
        in_n = cols < total
        k = tl.load(K + cols[:, None] * skt + kh * skh + d[None, :], mask=in_n[:, None], other=0.0)
        v = tl.load(V + cols[:, None] * svt + kh * svh + d[None, :], mask=in_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * c
        keep = (cols[None, :] >= st[:, None]) & (cols[None, :] < en[:, None])
        if window > 0:
            dist = rows[:, None] - cols[None, :]
            keep &= (dist < window) & (dist > -window)
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(
        OUT + rows[:, None] * sot + h * soh + d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=in_m[:, None],
    )


@triton.jit
def segment_mean(X, OUT, SEG_START, SEG_LEN, sxt, C, BT: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    in_c = cols < C
    first = tl.load(SEG_START + n)
    length = tl.load(SEG_LEN + n)
    acc = tl.zeros((BC,), tl.float32)
    for t in range(0, length, BT):
        rows = first + t + tl.arange(0, BT)
        keep = ((t + tl.arange(0, BT)) < length)[:, None] & in_c[None, :]
        x = tl.load(X + rows[:, None] * sxt + cols[None, :], mask=keep, other=0.0)
        acc += tl.sum(x.to(tl.float32), axis=0)
    mean = acc / tl.maximum(length, 1).to(tl.float32)
    tl.store(OUT + n * C + cols, mean, mask=in_c)
