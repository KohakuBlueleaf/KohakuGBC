"""Packed variable-length ops, eager: sequences laid end to end in one token stream, each
token carrying its sequence's span (``start`` / ``end``, ``[T]`` int32).

- ``varlen_attention(q, k, v, start, end, window, scale)``: bidirectional attention of each
  token over its own sequence (``q`` ``[T, HQ, D]``, ``k`` / ``v`` ``[T, HK, D]``, HQ a
  multiple of HK: grouped heads); ``window`` > 0 keeps the keys at distance < ``window``.
  Query rows are taken ``CHUNK`` at a time against the whole stream with a dense mask (no host
  sync, so the op runs inside a CUDA graph).
- ``segment_mean(x, seg_start, seg_len)``: the fp32 mean of each segment's rows of ``x``
  ``[T, C]`` (``seg_*`` ``[N]``; a zero-length segment gives zeros).
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry

CHUNK = 1024


def varlen_attention(q, k, v, start, end, window: int, scale: float) -> torch.Tensor:
    total, heads, _ = q.shape
    group = heads // k.shape[1]
    kt = k.repeat_interleave(group, dim=1).transpose(0, 1)[None]
    vt = v.repeat_interleave(group, dim=1).transpose(0, 1)[None]
    cols = torch.arange(total, device=q.device)
    out = []
    for lo in range(0, total, CHUNK):
        rows = cols[lo : lo + CHUNK]
        mask = (cols[None, :] >= start[rows, None]) & (cols[None, :] < end[rows, None])
        if window > 0:
            mask &= (rows[:, None] - cols[None, :]).abs() < window
        qt = q[lo : lo + CHUNK].transpose(0, 1)[None]
        o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask[None, None], scale=scale)
        out.append(o[0].transpose(0, 1))
    return torch.cat(out)


def segment_mean(x: torch.Tensor, seg_start, seg_len) -> torch.Tensor:
    total = x.shape[0]
    cols = torch.arange(total, device=x.device)
    member = (cols[None, :] >= seg_start[:, None]) & (
        cols[None, :] < (seg_start + seg_len)[:, None]
    )
    sums = member.to(torch.float32) @ x.float()
    return sums / seg_len.clamp(min=1)[:, None].to(torch.float32)


registry.register("varlen_attention", "eager", varlen_attention)
registry.register("segment_mean", "eager", segment_mean)
