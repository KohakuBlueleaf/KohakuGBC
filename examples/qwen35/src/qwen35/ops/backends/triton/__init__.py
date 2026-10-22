"""Triton backends written for this engine (slot-indexed recurrent delta rule, paged
attention)."""

from qwen35.ops.backends.triton import attention, delta

__all__ = ["attention", "delta"]
