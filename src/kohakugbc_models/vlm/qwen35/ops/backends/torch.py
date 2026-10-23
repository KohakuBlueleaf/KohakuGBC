"""PyTorch backends: the convolution vectorised over every sequence (no host sync, so it runs
inside a captured decode graph), and attention through ``scaled_dot_product_attention`` per
sequence (prefill)."""

import torch
import torch.nn.functional as F

from kohakugbc_models.vlm.qwen35.ops import register
from kohakugbc_models.vlm.qwen35.ops.backends.reference import gather_rows, spans, write_cache


@register("conv", "torch", graph_safe=True)
def conv(x, weight, state, slots, cu):
    """Each sequence's inputs laid after its ``width - 1`` saved ones in one packed buffer; the
    output of a token sums ``width`` shifted rows of it."""
    n, channels = x.shape
    width = weight.shape[1]
    keep = width - 1
    seqs = slots.shape[0]
    dev = x.device
    cu = cu.long()
    seg = torch.searchsorted(cu[1:], torch.arange(n, device=dev), right=True)
    ext = x.new_empty(n + seqs * keep, channels)
    rows = torch.arange(n, device=dev) + (seg + 1) * keep
    ext[rows] = x
    saved = (cu[:-1] + torch.arange(seqs, device=dev) * keep)[:, None] + torch.arange(
        keep, device=dev
    )
    ext[saved.flatten()] = state[slots].transpose(1, 2).reshape(-1, channels).to(x.dtype)
    w = weight.float()
    acc = torch.zeros(n, channels, dtype=torch.float32, device=dev)
    for j in range(width):
        acc += ext[rows - (keep - j)].float() * w[:, j]
    last = (cu[1:] + (torch.arange(seqs, device=dev) + 1) * keep)[:, None] - keep
    last = last + torch.arange(keep, device=dev)
    state[slots] = ext[last.flatten()].view(seqs, keep, channels).transpose(1, 2).to(state.dtype)
    return F.silu(acc).to(x.dtype)


@register("attention", "sdpa")
def attention(q, k, v, cache_k, cache_v, pages, ctx, cu, write, max_len, scale):
    write_cache(k, v, cache_k, cache_v, write)
    page, kvh, hd = cache_k.shape[1], cache_k.shape[2], cache_k.shape[3]
    out = torch.empty_like(q)
    lengths = ctx.tolist()
    for n, (s, e) in enumerate(spans(cu)):
        length, new = lengths[n], e - s
        rows = gather_rows(pages[n], page, length)
        keys = cache_k.view(-1, kvh, hd)[rows].transpose(0, 1)[None]
        values = cache_v.view(-1, kvh, hd)[rows].transpose(0, 1)[None]
        query = q[s:e].transpose(0, 1)[None]
        if new == length:
            o = F.scaled_dot_product_attention(
                query, keys, values, is_causal=True, scale=scale, enable_gqa=True
            )
        else:
            position = length - new + torch.arange(new, device=q.device)
            mask = torch.arange(length, device=q.device)[None, :] <= position[:, None]
            o = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, scale=scale, enable_gqa=True
            )
        out[s:e] = o[0].transpose(0, 1)
    return out
