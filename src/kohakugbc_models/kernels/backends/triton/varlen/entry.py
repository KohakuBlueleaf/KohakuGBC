"""The Triton packed variable-length ops and their per-problem choice (``choose_in``: every
tile against the PyTorch paths, the choice bound per problem key and tuning metric)."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import varlen as eager
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.varlen._kernels import segment_mean as segment_kernel
from kohakugbc_models.kernels.backends.triton.varlen._kernels import varlen_fwd

DTYPES = (torch.bfloat16, torch.float16)
TILES = [
    {"BM": bm, "BN": bn, "num_warps": w, "num_stages": s}
    for bm, bn, w, s in (
        (32, 32, 4, 2),
        (32, 64, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 8, 2),
        (16, 32, 4, 2),
        (64, 32, 8, 1),
    )
]
MEAN_TILES = [{"BT": bt, "BC": bc, "num_warps": w} for bt, bc, w in ((16, 256, 4), (32, 128, 4))]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def pytorch_path(op: str, name: str):
    ref = getattr(eager, op)
    return ref if name == "torch" else compiled(op, ref)


def varlen_attention(q, k, v, start, end, window: int, scale: float) -> torch.Tensor:
    total, heads, dim = q.shape
    out = torch.empty_like(q)

    def tile(config):
        grid = (triton.cdiv(total, config["BM"]), heads)

        def run():
            varlen_fwd[grid](
                q,
                k,
                v,
                out,
                start,
                end,
                *q.stride()[:2],
                *k.stride()[:2],
                *v.stride()[:2],
                *out.stride()[:2],
                total,
                window,
                scale,
                GROUP=heads // k.shape[1],
                D=dim,
                BM=config["BM"],
                BN=config["BN"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            return out

        return run

    def path(name):
        fn = pytorch_path("varlen_attention", name)
        return lambda: fn(q, k, v, start, end, window, scale)

    key = f"{q.dtype}-h{heads}-g{heads // k.shape[1]}-d{dim}-w{int(window > 0)}-t{bucket(total)}"
    config = choose_in("varlen_attention", key, q.device, tile, path, TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


def segment_mean(x, seg_start, seg_len) -> torch.Tensor:
    total, channels = x.shape
    segments = seg_start.shape[0]
    out = torch.empty(segments, channels, dtype=torch.float32, device=x.device)

    def tile(config):
        grid = (segments, triton.cdiv(channels, config["BC"]))

        def run():
            segment_kernel[grid](
                x,
                out,
                seg_start,
                seg_len,
                x.stride(0),
                channels,
                BT=config["BT"],
                BC=config["BC"],
                num_warps=config["num_warps"],
            )
            return out

        return run

    def path(name):
        fn = pytorch_path("segment_mean", name)
        return lambda: fn(x, seg_start, seg_len)

    key = f"{x.dtype}-c{channels}-n{segments}-t{bucket(total)}"
    config = choose_in("segment_mean", key, x.device, tile, path, MEAN_TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


registry.register("varlen_attention", "triton", varlen_attention, supports)
registry.register("segment_mean", "triton", segment_mean)
