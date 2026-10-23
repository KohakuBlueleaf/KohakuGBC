"""Triton block ops: ``linear_act`` (GEMM + activation epilogue), ``qkv_rope`` (the ViT's fused
q / k / v GEMM + rotary epilogue) and ``add_layer_norm``; 16-bit operands only."""
