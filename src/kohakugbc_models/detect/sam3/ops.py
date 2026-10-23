"""The ops SAM 3's layers call, bound per model to its device and dtype.

Each layer that calls ops holds an ``Ops``: eager by default (the same casts and operation
order as transformers' SAM 3, so the eager model is the reference model), or resolved through
the kernel registry (``Ops.resolved``) so each op runs the backend the arch's priority and its
fallbacks give. The ops are those of ``kohakugbc_models.kernels.backends.eager``: ``attention``,
``vit_attention``, ``attention_rpb`` (``attention``), ``linear_act``, ``qkv_rope`` and
``add_layer_norm`` (``blocks``), ``box_bias`` (``box_bias``), ``quant``, ``add_ln_quant`` and
``act_quant`` (``quant``: the fp8 producers), ``mask_box`` (``mask_box``).
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.eager import attention as eager_attention
from kohakugbc_models.kernels.backends.eager import blocks as eager_blocks
from kohakugbc_models.kernels.backends.eager import box_bias as eager_box_bias
from kohakugbc_models.kernels.backends.eager import mask_box as eager_mask_box
from kohakugbc_models.kernels.backends.eager import quant as eager_quant

EAGER_IMPLS = {
    "attention": eager_attention.attention,
    "vit_attention": eager_attention.vit_attention,
    "attention_rpb": eager_attention.attention_rpb,
    "linear_act": eager_blocks.linear_act,
    "qkv_rope": eager_blocks.qkv_rope,
    "add_layer_norm": eager_blocks.add_layer_norm,
    "box_bias": eager_box_bias.box_bias,
    "quant": eager_quant.quant,
    "add_ln_quant": eager_quant.add_ln_quant,
    "act_quant": eager_quant.act_quant,
    "mask_box": eager_mask_box.mask_box,
}
NAMES = tuple(EAGER_IMPLS)


class Ops:
    def __init__(self, **bound):
        for name in NAMES:
            setattr(self, name, bound.get(name, EAGER_IMPLS[name]))

    @classmethod
    def resolved(cls, device, dtype: torch.dtype) -> "Ops":
        return cls(**{name: registry.resolve(name, device, dtype) for name in NAMES})

    def backends(self) -> dict:
        return {n: getattr(getattr(self, n), "backend", "eager") for n in NAMES}


EAGER = Ops()
layer_norm = F.layer_norm
linear = F.linear
gelu = F.gelu
