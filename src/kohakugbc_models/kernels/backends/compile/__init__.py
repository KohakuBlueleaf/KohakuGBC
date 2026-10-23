"""torch.compile of every eager reference, as a backend of its own: on a GPU its kernels are
Triton (it probes as Triton does), on the host it needs a C++ compiler (probed by compiling a
one-op function once per device).

Each op is compiled once per process (``dynamic=False``: one graph per shape, as the kernels
are keyed by shape too). The ops meet one shape per problem, more than Dynamo's default
recompile limit (8), past which it would silently run eager: the limit is raised to
``RECOMPILE_LIMIT`` when this backend loads."""

import torch
import torch._dynamo

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends import load_tree
from kohakugbc_models.kernels.backends.eager import attention as eager_attention
from kohakugbc_models.kernels.backends.eager import blocks as eager_blocks
from kohakugbc_models.kernels.backends.eager import box_bias as eager_box_bias
from kohakugbc_models.kernels.backends.eager import mask_box as eager_mask_box
from kohakugbc_models.kernels.backends.eager import quant as eager_quant
from kohakugbc_models.kernels.backends.eager import varlen as eager_varlen

_COMPILED: dict[str, object] = {}
RECOMPILE_LIMIT = 256
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
torch._dynamo.config.accumulated_recompile_limit = max(
    torch._dynamo.config.accumulated_recompile_limit, 4 * RECOMPILE_LIMIT
)


def compiled(op: str, fn):
    if op not in _COMPILED:
        _COMPILED[op] = torch.compile(fn, dynamic=False)
    return _COMPILED[op]


def probe(device: str) -> str | None:
    if torch.device(device).type == "cuda":
        return registry.probe("triton", device) if "triton" in registry.PROBES else None
    try:
        fn = torch.compile(lambda x: x * 2 + 1, dynamic=False)
        fn(torch.ones(4, device=device))
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return None


registry.register_probe("compile", probe)
for _op in ("attention", "vit_attention", "attention_rpb"):
    registry.register(_op, "compile", compiled(_op, getattr(eager_attention, _op)))
for _op in ("linear_act", "qkv_rope", "add_layer_norm"):
    registry.register(_op, "compile", compiled(_op, getattr(eager_blocks, _op)))
registry.register("box_bias", "compile", compiled("box_bias", eager_box_bias.box_bias))
registry.register("mask_box", "compile", compiled("mask_box", eager_mask_box.mask_box))
for _op in ("quant", "add_ln_quant", "act_quant"):
    registry.register(_op, "compile", compiled(_op, getattr(eager_quant, _op)))
for _op in ("varlen_attention", "segment_mean"):
    registry.register(_op, "compile", compiled(_op, getattr(eager_varlen, _op)))

FAILED = load_tree(__name__, __path__)
