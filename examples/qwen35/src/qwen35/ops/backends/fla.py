"""flash-linear-attention's chunked gated delta rule (Triton): prefill from each sequence's
saved state, the final states written back to the pool."""

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from qwen35.ops import register
from qwen35.ops.backends.reference import gates


@register("delta", "fla")
def delta(q, k, v, a, b, a_log, dt_bias, state, slots, cu, max_len):
    g, beta = gates(a, b, a_log, dt_bias)
    o, final = chunk_gated_delta_rule(
        q[None],
        k[None],
        v[None],
        g[None],
        beta.to(q.dtype)[None],
        scale=q.shape[-1] ** -0.5,
        initial_state=state[slots].float(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu.long(),
    )
    state[slots] = final.to(state.dtype)
    return o[0]
