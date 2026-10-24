"""The ``qwen35`` tests need the ``qwen35`` extra (torch, triton, flash-linear-attention,
xgrammar); without it they are not collected."""

from importlib.util import find_spec

QWEN35_EXTRA = ("torch", "triton", "fla", "xgrammar")

collect_ignore = [] if all(find_spec(m) for m in QWEN35_EXTRA) else ["qwen35"]
