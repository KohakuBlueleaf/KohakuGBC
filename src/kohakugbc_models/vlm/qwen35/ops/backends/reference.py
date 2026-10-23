"""The eager references: one sequence and one token at a time, in fp32, rounded once to the
input dtype. Every backend is checked against these (``tests/test_ops.py``)."""

import torch
import torch.nn.functional as F

from kohakugbc_models.vlm.qwen35.ops import register

L2_EPS = 1e-6


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + L2_EPS)


def gates(a, b, a_log, dt_bias) -> tuple[torch.Tensor, torch.Tensor]:
    """The log decay ``g`` and the write strength ``beta`` of every token and value head."""
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, b.float().sigmoid()


def spans(cu: torch.Tensor) -> list[tuple[int, int]]:
    bounds = cu.tolist()
    return list(zip(bounds[:-1], bounds[1:]))


@register("delta", "eager")
def delta(q, k, v, a, b, a_log, dt_bias, state, slots, cu, max_len):
    groups = v.shape[1] // q.shape[1]
    g, beta = gates(a, b, a_log, dt_bias)
    qf = l2norm(q.float()).repeat_interleave(groups, 1) * q.shape[-1] ** -0.5
    kf = l2norm(k.float()).repeat_interleave(groups, 1)
    vf = v.float()
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    for n, (s, e) in enumerate(spans(cu)):
        slot = int(slots[n])
        h = state[slot].float()
        for t in range(s, e):
            h = h * g[t].exp()[:, None, None]
            seen = (h * kf[t][:, :, None]).sum(1)
            dv = (vf[t] - seen) * beta[t][:, None]
            h = h + kf[t][:, :, None] * dv[:, None, :]
            out[t] = (h * qf[t][:, :, None]).sum(1)
        state[slot] = h.to(state.dtype)
    return out.to(v.dtype)


@register("conv", "eager")
def conv(x, weight, state, slots, cu):
    width = weight.shape[1]
    w = weight.float().t()
    out = torch.empty_like(x)
    for n, (s, e) in enumerate(spans(cu)):
        slot = int(slots[n])
        ext = torch.cat([state[slot].float().t(), x[s:e].float()])
        for t in range(e - s):
            out[s + t] = F.silu((ext[t : t + width] * w).sum(0)).to(x.dtype)
        state[slot] = ext[ext.shape[0] - (width - 1) :].t().to(state.dtype)
    return out


def write_cache(k, v, cache_k, cache_v, write) -> None:
    kvh, hd = k.shape[1], k.shape[2]
    cache_k.view(-1, kvh, hd)[write] = k
    cache_v.view(-1, kvh, hd)[write] = v


def gather_rows(pages: torch.Tensor, page: int, length: int) -> torch.Tensor:
    """Flat cache rows of a sequence's first ``length`` tokens through its page table."""
    used = pages[: -(-length // page)].long()
    rows = used[:, None] * page + torch.arange(page, device=pages.device)
    return rows.flatten()[:length]


@register("attention", "eager")
def attention(q, k, v, cache_k, cache_v, pages, ctx, cu, write, max_len, scale):
    write_cache(k, v, cache_k, cache_v, write)
    page, kvh, hd = cache_k.shape[1], cache_k.shape[2], cache_k.shape[3]
    groups = q.shape[1] // kvh
    out = torch.empty_like(q)
    lengths = ctx.tolist()
    for n, (s, e) in enumerate(spans(cu)):
        length = lengths[n]
        rows = gather_rows(pages[n], page, length)
        keys = cache_k.view(-1, kvh, hd)[rows].float().repeat_interleave(groups, 1)
        values = cache_v.view(-1, kvh, hd)[rows].float().repeat_interleave(groups, 1)
        scores = torch.einsum("thd,lhd->htl", q[s:e].float(), keys) * scale
        position = length - (e - s) + torch.arange(e - s, device=q.device)
        visible = torch.arange(length, device=q.device)[None, :] <= position[:, None]
        scores = scores.masked_fill(~visible[None], float("-inf"))
        out[s:e] = torch.einsum("htl,lhd->thd", scores.softmax(-1), values).to(q.dtype)
    return out
