"""Every backend, registered on import; one whose library is missing is recorded in
``qwen35.ops.UNAVAILABLE`` and never a candidate."""

from qwen35.ops import UNAVAILABLE
from qwen35.ops.backends import reference
from qwen35.ops.backends import torch as torch_ops

try:
    from qwen35.ops.backends import fla
except ImportError as exc:
    fla = None
    UNAVAILABLE[("delta", "fla")] = repr(exc)
try:
    from qwen35.ops.backends import triton
except ImportError as exc:
    triton = None
    UNAVAILABLE[("delta", "triton")] = UNAVAILABLE[("attention", "triton")] = repr(exc)

__all__ = ["fla", "reference", "torch_ops", "triton"]
