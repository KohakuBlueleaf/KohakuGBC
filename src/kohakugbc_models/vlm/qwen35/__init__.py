"""A specialised Qwen3.5 (2B / 4B / 9B) VLM engine for KohakuGBC's ``vlm`` slot.

``qwen35.model`` holds the checkpoint's shape and forward, ``qwen35.ops`` the op registry and
its backends, ``qwen35.serve`` the pools, inputs, grammar, graphs and the engine loop, and
``qwen35.unit`` the Togeari unit and the ``qwen35`` entry.
"""
