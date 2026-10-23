"""A Qwen3.5 (2B / 4B / 9B) VLM engine written for GBC's workload.

``model`` holds the checkpoint's shape and forward, ``ops`` the op registry and its backends,
``serve`` the pools, inputs, grammar, graphs and the engine loop; the Togeari unit and the
``qwen35`` entry of the ``vlm`` slot are ``kohakugbc.engines.vlm.qwen35``.
"""
