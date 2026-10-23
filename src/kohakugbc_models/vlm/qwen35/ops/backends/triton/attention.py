"""Paged causal attention over any mix of sequences (decode tokens, short extends, prefill
chunks): one program per (block of ``BQ`` new tokens of one sequence, kv head) computes every
query head of the group for those tokens at once (``BQ * group`` rows, padded to the 16 a
``tl.dot`` needs), reading the sequence's cached keys and values through its page table with an
online softmax. Decode-only calls use ``BQ = 1``; others ``BQ = 16``.

The block map (sequence and first token of every block) is built on the device from ``cu`` with
a grid of ``ceil(N / BQ) + B`` programs, the unused ones returning at once: no host sync, so the
op runs inside a captured graph.
"""

import torch
import triton
import triton.language as tl

from qwen35.ops import register
from qwen35.ops.backends.reference import write_cache


@triton.jit
def _paged_attention(
    Q,
    KC,
    VC,
    O,
    PAGES,
    CU,
    CTX,
    SEG,
    QST,
    scale,
    n_seq,
    s_qn,
    s_qh,
    s_cp,
    s_co,
    s_ch,
    s_pb,
    s_on,
    s_oh,
    GROUP: tl.constexpr,
    BQ: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    n = tl.load(SEG + pid)
    if n >= n_seq:
        return
    n = n.to(tl.int64)
    q0 = tl.load(QST + pid)
    q_end = tl.load(CU + n + 1)
    ctx = tl.load(CTX + n)
    r = tl.arange(0, M)
    qi = q0 + r // GROUP
    head = kh * GROUP + r % GROUP
    live = (r < BQ * GROUP) & (qi < q_end)
    pos = tl.where(live, ctx - (q_end - qi), 0)
    last = tl.max(pos, 0)
    d = tl.arange(0, D)
    q = tl.load(
        Q + qi.to(tl.int64)[:, None] * s_qn + head[:, None] * s_qh + d[None, :],
        mask=live[:, None],
        other=0.0,
    )
    m = tl.full([M], float("-inf"), dtype=tl.float32)
    total = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)
    for start in range(0, last + 1, BN):
        j = start + tl.arange(0, BN)
        inside = j <= last
        page = tl.load(PAGES + n * s_pb + j // PAGE, mask=inside, other=0).to(tl.int64)
        off = page * s_cp + (j % PAGE) * s_co + kh * s_ch
        keys = tl.load(KC + off[:, None] + d[None, :], mask=inside[:, None], other=0.0)
        s = tl.dot(q, tl.trans(keys)) * scale
        s = tl.where(j[None, :] <= pos[:, None], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        total = total * alpha + tl.sum(p, 1)
        values = tl.load(VC + off[:, None] + d[None, :], mask=inside[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(values.dtype), values)
        m = m_new
    out = acc / total[:, None]
    tl.store(
        O + qi.to(tl.int64)[:, None] * s_on + head[:, None] * s_oh + d[None, :],
        out.to(O.dtype.element_ty),
        mask=live[:, None],
    )


def block_map(cu: torch.Tensor, n: int, bq: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(sequence, first token) of every block of ``bq`` new tokens, over a grid of
    ``ceil(n / bq) + B`` blocks; unused blocks name sequence ``B``."""
    seqs = cu.shape[0] - 1
    grid = -(-n // bq) + seqs
    cu = cu.long()
    lens = cu[1:] - cu[:-1]
    blocks = (lens + bq - 1) // bq
    ends = torch.cumsum(blocks, 0)
    ids = torch.arange(grid, device=cu.device)
    seg = torch.searchsorted(ends, ids, right=True)
    safe = seg.clamp_max(seqs - 1)
    first = cu[safe] + (ids - (ends[safe] - blocks[safe])) * bq
    return seg.to(torch.int32), first.to(torch.int32), grid


@register("attention", "triton", graph_safe=True)
def attention(q, k, v, cache_k, cache_v, pages, ctx, cu, write, max_len, scale):
    write_cache(k, v, cache_k, cache_v, write)
    n, heads, dim = q.shape
    kvh, page = cache_k.shape[2], cache_k.shape[1]
    group = heads // kvh
    bq = 1 if max_len == 1 else 16
    rows = max(16, triton.next_power_of_2(bq * group))
    seg, first, grid = block_map(cu, n, bq)
    out = torch.empty_like(q)
    _paged_attention[(grid, kvh)](
        q,
        cache_k,
        cache_v,
        out,
        pages,
        cu,
        ctx,
        seg,
        first,
        scale,
        cu.shape[0] - 1,
        q.stride(0),
        q.stride(1),
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        pages.stride(0),
        out.stride(0),
        out.stride(1),
        GROUP=group,
        BQ=bq,
        M=rows,
        D=dim,
        PAGE=page,
        BN=32,
        num_warps=4 if rows <= 16 else 8,
        num_stages=2,
    )
    return out
