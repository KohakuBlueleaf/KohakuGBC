"""Every backend, registered on import; one whose library is missing is recorded in
``kohakugbc_models.vlm.qwen35.ops.UNAVAILABLE`` and never a candidate."""

from kohakugbc_models.vlm.qwen35.ops import UNAVAILABLE
from kohakugbc_models.vlm.qwen35.ops.backends import reference
from kohakugbc_models.vlm.qwen35.ops.backends import torch as torch_ops

try:
    from kohakugbc_models.vlm.qwen35.ops.backends import fla
except ImportError as exc:
    fla = None
    UNAVAILABLE[("delta", "fla")] = repr(exc)
try:
    from kohakugbc_models.vlm.qwen35.ops.backends import triton
except ImportError as exc:
    triton = None
    UNAVAILABLE[("delta", "triton")] = UNAVAILABLE[("attention", "triton")] = repr(exc)

__all__ = ["fla", "reference", "torch_ops", "triton"]
