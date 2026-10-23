"""The Triton ``box_bias`` and its per-problem choice (``choose_in``: the configs against the
PyTorch paths)."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import box_bias as eager
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.boxbias._kernels import axis_table

DTYPES = (torch.bfloat16, torch.float16)
TILES = [{"BK": bk, "num_warps": w} for bk in (32, 64) for w in (4, 8)]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def box_bias(boxes, y_w1, y_b1, y_w2, y_b2, x_w1, x_b1, x_w2, x_b2, height, width) -> tuple:
    y_mlp, x_mlp = (y_w1, y_b1, y_w2, y_b2), (x_w1, x_b1, x_w2, x_b2)
    boxes = boxes.contiguous()
    batch, queries, _ = boxes.shape
    heads, hidden = y_mlp[2].shape
    shape = (batch, heads, 1 + queries)
    y = torch.zeros(*shape, height, dtype=boxes.dtype, device=boxes.device)
    x = torch.zeros(*shape, width, dtype=boxes.dtype, device=boxes.device)

    def tile(config):
        def run():
            for axis, table, mlp, size in ((1, y, y_mlp, height), (0, x, x_mlp, width)):
                w1, b1, w2, b2 = mlp
                axis_table[(batch, queries)](
                    boxes,
                    w1,
                    b1,
                    w2,
                    b2,
                    table,
                    boxes.stride(0),
                    boxes.stride(1),
                    *table.stride()[:3],
                    size,
                    AXIS=axis,
                    HID=hidden,
                    HEADS=heads,
                    NH=max(16, triton.next_power_of_2(heads)),
                    BG=triton.next_power_of_2(size),
                    BK=config["BK"],
                    num_warps=config["num_warps"],
                )
            return y, x

        return run

    def path(name):
        fn = eager.box_bias if name == "torch" else compiled("box_bias", eager.box_bias)
        return lambda: fn(boxes, *y_mlp, *x_mlp, height, width)

    key = f"{boxes.dtype}-q{queries}-h{heads}-k{hidden}-g{height}x{width}-b{bucket(batch)}"
    config = choose_in("box_bias", key, boxes.device, tile, path, TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


registry.register("box_bias", "triton", box_bias, supports)
