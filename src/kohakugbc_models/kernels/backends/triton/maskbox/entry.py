"""The Triton ``mask_box`` and its per-problem choice (``choose_in``: tiles against the PyTorch
paths). The instance projection folds into the queries here (two small GEMMs); the kernel
then reads the pixel embeddings once per query block and writes only the boxes."""

import math

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import mask_box as eager
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.maskbox._kernels import mask_box_fwd
from kohakugbc_models.kernels.hardware import tiles_for

DTYPES = (torch.bfloat16, torch.float16)
TILES = [
    {"BQ": bq, "BP": bp, "BC": bc, "num_warps": w, "num_stages": s}
    for bq, bp, bc, w, s in (
        (64, 64, 64, 4, 3),
        (64, 128, 64, 8, 3),
        (128, 64, 64, 8, 3),
        (32, 128, 64, 4, 3),
        (64, 128, 32, 4, 4),
    )
]
# sm10x: larger query / pixel blocks with deeper pipelines (228 KiB of shared memory per SM)
EXTRA = {
    "sm10x": [
        {"BQ": bq, "BP": bp, "BC": bc, "num_warps": w, "num_stages": s}
        for bq, bp, bc, w, s in ((128, 128, 64, 8, 3), (128, 128, 64, 8, 4), (64, 256, 64, 8, 3))
    ]
}
PYTORCH = [{"path": "torch"}, {"path": "compile"}]
SPLIT = True  # the folded queries as hi + lo 16-bit parts (see _kernels)


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def mask_box(emb, pixels, weight, bias, threshold: float) -> tuple:
    batch, queries, channels = emb.shape
    height, width = pixels.shape[2:]
    folded32 = emb.float() @ weight.view(channels, channels).float()
    folded = folded32.to(emb.dtype).contiguous()
    folded_lo = (folded32 - folded.float()).to(emb.dtype).contiguous()
    if not SPLIT:
        folded_lo = torch.zeros_like(folded)
    qb = (emb.float() @ bias.float()).contiguous()
    pix = pixels.reshape(batch, channels, height * width)
    logit = math.log(threshold / (1.0 - threshold)) if 0.0 < threshold < 1.0 else threshold

    def tile(config):
        def run():
            box = torch.tensor([width, height, 0, 0], dtype=torch.int32, device=emb.device)
            box = box.repeat(batch, queries, 1).contiguous()
            grid = (
                batch,
                triton.cdiv(queries, config["BQ"]),
                triton.cdiv(height * width, config["BP"]),
            )
            mask_box_fwd[grid](
                folded,
                folded_lo,
                qb,
                pix,
                box,
                logit,
                queries,
                channels,
                height * width,
                width,
                folded.stride(0),
                folded.stride(1),
                pix.stride(0),
                pix.stride(1),
                box.stride(0),
                box.stride(1),
                BQ=config["BQ"],
                BP=config["BP"],
                BC=config["BC"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            empty = box[..., 3] == 0
            # An empty mask's box is the whole grid, as the eager argmax over no kept pixel gives
            whole = torch.tensor([0, 0, width, height], dtype=torch.int32, device=emb.device)
            return torch.where(empty[..., None], whole, box), empty

        return run

    def path(name):
        fn = eager.mask_box if name == "torch" else compiled("mask_box", eager.mask_box)
        return lambda: fn(emb, pixels, weight, bias, threshold)

    key = f"{emb.dtype}-q{queries}-c{channels}-g{height}x{width}-b{bucket(batch)}"
    tiles = tiles_for(emb.device, TILES, EXTRA)
    config = choose_in("mask_box", key, emb.device, tile, path, tiles + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


registry.register("mask_box", "triton", mask_box, supports)
