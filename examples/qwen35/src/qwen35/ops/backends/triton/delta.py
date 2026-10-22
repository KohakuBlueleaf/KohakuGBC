"""The gated delta rule token by token, with each sequence's state read from and written back to
its pool slot in place (no gather or scatter of states), the gates computed in the kernel.
Decode and short extends (forced tokens) run here; one program per (value block, sequence,
value head)."""

import torch
import triton
import triton.language as tl

from qwen35.ops import register


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _delta_recurrent(
    Q, K, V, A, B, ALOG, DTB, O, STATE, SLOTS, CU, scale,
    H: tl.constexpr, HV: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, BV: tl.constexpr,
):  # fmt: skip
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    n = i_nh // HV
    hv = i_nh % HV
    h = hv // (HV // H)
    bos = tl.load(CU + n).to(tl.int64)
    eos = tl.load(CU + n + 1).to(tl.int64)
    slot = tl.load(SLOTS + n).to(tl.int64)
    o_k = tl.arange(0, KD)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = STATE + (slot * HV + hv) * KD * VD + o_k[:, None] * VD + o_v[None, :]
    b_h = tl.load(p_h).to(tl.float32)
    a_log = tl.load(ALOG + hv).to(tl.float32)
    dt_bias = tl.load(DTB + hv).to(tl.float32)
    for t in range(bos, eos):
        b_q = tl.load(Q + (t * H + h) * KD + o_k).to(tl.float32)
        b_k = tl.load(K + (t * H + h) * KD + o_k).to(tl.float32)
        b_v = tl.load(V + (t * HV + hv) * VD + o_v).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        g = -tl.exp(a_log) * _softplus(tl.load(A + t * HV + hv).to(tl.float32) + dt_bias)
        beta = tl.sigmoid(tl.load(B + t * HV + hv).to(tl.float32))
        b_h *= tl.exp(g)
        b_v = beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(O + (t * HV + hv) * VD + o_v, b_o.to(O.dtype.element_ty))
    tl.store(p_h, b_h.to(STATE.dtype.element_ty))


@register("delta", "triton", graph_safe=True)
def delta(q, k, v, a, b, a_log, dt_bias, state, slots, cu, max_len):
    n_seq = slots.shape[0]
    hk, kd = q.shape[1], q.shape[2]
    hv, vd = v.shape[1], v.shape[2]
    bv = 8
    out = torch.empty_like(v)
    grid = (vd // bv, n_seq * hv)
    _delta_recurrent[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(),
        a_log, dt_bias, out, state, slots, cu, kd ** -0.5,
        H=hk, HV=hv, KD=kd, VD=vd, BV=bv, num_warps=1, num_stages=3,
    )  # fmt: skip
    return out
