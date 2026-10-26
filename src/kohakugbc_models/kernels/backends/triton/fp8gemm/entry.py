"""``rowwise_mm``: the Triton fp8 row-wise GEMM at its tuned tile (tiles timed per problem)."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.fp8gemm import _kernels
from kohakugbc_models.kernels.hardware import tiles_for

TILES = [
    {"BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s, "GROUP": 8}
    for bm, bn, bk, w, s in (
        (128, 128, 64, 8, 3),
        (128, 128, 64, 4, 4),
        (128, 256, 64, 8, 3),
        (64, 128, 64, 4, 4),
        (128, 64, 128, 4, 3),
        (64, 64, 128, 4, 3),
    )
]
# sm10x: 256-wide tiles with K 128 (228 KiB of shared memory per SM)
EXTRA = {
    "sm10x": [
        {"BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s, "GROUP": 8}
        for bm, bn, bk, w, s in ((128, 256, 128, 8, 3), (256, 128, 128, 8, 3), (256, 256, 64, 8, 3))
    ]
}


def rowwise_mm(xq, xs, wq, ws, bias=None, out_dtype=torch.bfloat16) -> torch.Tensor:
    """``xq [M, K] @ wq[N, K]^T`` scaled by ``xs [M, 1]`` / ``ws [1, N]`` (+ ``bias``)."""
    m, k = xq.shape
    n = wq.shape[0]
    out = torch.empty(m, n, dtype=out_dtype, device=xq.device)
    sa, sb = xs.reshape(-1).contiguous(), ws.reshape(-1).contiguous()

    def tile(config):
        grid = (triton.cdiv(m, config["BM"]) * triton.cdiv(n, config["BN"]),)

        def run():
            _kernels.rowwise_mm[grid](
                xq,
                wq,
                sa,
                sb,
                bias if bias is not None else sa,
                out,
                m,
                n,
                k,
                xq.stride(0),
                wq.stride(0),
                out.stride(0),
                HAS_BIAS=bias is not None,
                BM=config["BM"],
                BN=config["BN"],
                BK=config["BK"],
                GROUP=config["GROUP"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            return out

        return run

    key = f"m{bucket(m)}-n{n}-k{k}-{out_dtype}"
    config = choose_in(
        "fp8_rowwise_mm", key, xq.device, tile, None, tiles_for(xq.device, TILES, EXTRA)
    )
    return tile(config)()


registry.register("fp8_rowwise_mm", "triton", rowwise_mm)
