"""The Triton block ops and their per-problem choice.

As the attention ops: each problem (op, dtype, shape, batch bucket) picks, through
``tuning.select``, among the Triton configs and the PyTorch paths (``torch``: the eager
reference; ``compile``: torch.compile of it). ``qkv_rope`` has a third path, ``split``: one
cuBLAS GEMM over the fused weight, then ``vit_rope`` rotating q and k in place.
"""

import torch
import torch.nn.functional as F
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import blocks as eager
from kohakugbc_models.kernels.backends.triton.attention._kernels import vit_rope
from kohakugbc_models.kernels.backends.triton.attention.entry import ROPE_ROWS, bucket, choose_in
from kohakugbc_models.kernels.backends.triton.blocks import _kernels
from kohakugbc_models.kernels.hardware import tiles_for

DTYPES = (torch.bfloat16, torch.float16)
ACT = {"none": 0, "gelu": 1, "relu": 2}
GEMM_TILES = [
    {"BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s, "GROUP": 8}
    for bm, bn, bk, w, s in (
        (128, 128, 64, 8, 3),
        (128, 128, 64, 4, 3),
        (128, 128, 32, 4, 4),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 64, 64, 4, 4),
        (64, 128, 64, 4, 4),
        (64, 64, 64, 4, 4),
        (128, 128, 64, 8, 4),
        (64, 256, 32, 4, 4),
        (128, 128, 32, 4, 5),
        (128, 128, 32, 8, 4),
        (128, 256, 32, 8, 4),
        (256, 128, 32, 8, 4),
        (128, 64, 32, 4, 5),
        (64, 128, 32, 4, 5),
    )
]
# sm10x: 256-wide tiles with deeper pipelines (228 KiB of shared memory per SM)
GEMM_EXTRA = {
    "sm10x": [
        {"BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s, "GROUP": 8}
        for bm, bn, bk, w, s in (
            (256, 128, 64, 8, 4),
            (128, 256, 64, 8, 4),
            (256, 256, 64, 8, 3),
            (256, 256, 64, 16, 3),
            (128, 128, 128, 8, 3),
        )
    ]
}
NORM_TILES = [{"RB": rb, "num_warps": w} for rb in (1, 2, 4, 8, 16) for w in (1, 2, 4, 8)]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]


def pytorch_path(op: str, name: str):
    ref = getattr(eager, op)
    return ref if name == "torch" else compiled(op, ref)


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def launch_gemm(x2, w, b, out, cos, sin, config, act=0, rope=None) -> None:
    m, k = x2.shape
    n = w.shape[0]
    size, window, dim, channels = rope if rope is not None else (1, 1, 2, n)
    grid = (triton.cdiv(m, config["BM"]) * triton.cdiv(n, config["BN"]),)
    _kernels.gemm[grid](
        x2,
        w,
        b if b is not None else w,
        out,
        cos,
        sin,
        m,
        n,
        k,
        x2.stride(0),
        w.stride(0),
        n,
        channels,
        HAS_BIAS=b is not None,
        ACT=act,
        ROPE=rope is not None,
        SIZE=size,
        WIN=window,
        D=dim,
        BM=config["BM"],
        BN=config["BN"],
        BK=config["BK"],
        GROUP=config["GROUP"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def linear_act(x, w, b, act: str = "none") -> torch.Tensor:
    k, n = x.shape[-1], w.shape[0]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)

    def tile(config):
        def run():
            launch_gemm(x2, w, b, out, w, w, config, ACT[act])
            return out.view(*x.shape[:-1], n)

        return run

    def path(name):
        fn = pytorch_path("linear_act", name)
        return lambda: fn(x, w, b, act)

    key = f"{x.dtype}-m{bucket(m)}-n{n}-k{k}-{act}-b{int(b is not None)}"
    tiles = tiles_for(x.device, GEMM_TILES, GEMM_EXTRA)
    config = choose_in("linear_act", key, x.device, tile, path, tiles + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


def qkv_rope(x, w, b, cos, sin, window: int) -> tuple:
    batch, size, _, channels = x.shape
    dim = cos.shape[-1]
    heads = channels // dim
    length = window * window
    cos2, sin2 = cos.reshape(length, dim), sin.reshape(length, dim)
    x2 = x.reshape(-1, channels)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    shape = (batch, size, size, heads, dim)

    def tile(config):
        out = torch.empty(3, m, channels, dtype=x.dtype, device=x.device)

        def run():
            launch_gemm(x2, w, b, out, cos2, sin2, config, rope=(size, window, dim, channels))
            return tuple(out[i].view(shape) for i in range(3))

        return run

    def split():
        y = F.linear(x, w, b).view(batch, size, size, 3, heads, dim)
        q, k, v = y.unbind(3)
        rows = batch * size * size * heads
        vit_rope[(triton.cdiv(rows, ROPE_ROWS),)](
            q,
            k,
            q,
            k,
            cos2,
            sin2,
            rows,
            *q.stride()[1:4],
            size,
            WIN=window,
            HEADS=heads,
            D=dim,
            BR=ROPE_ROWS,
            num_warps=4,
        )
        return q, k, v

    def path(name):
        if name == "split":
            return split
        fn = pytorch_path("qkv_rope", name)
        return lambda: fn(x, w, b, cos, sin, window)

    candidates = tiles_for(x.device, GEMM_TILES, GEMM_EXTRA) + [{"path": "split"}] + PYTORCH
    key = f"{x.dtype}-c{channels}-d{dim}-s{size}-w{window}-b{bucket(batch)}"
    config = choose_in("qkv_rope", key, x.device, tile, path, candidates)
    return path(config["path"])() if "path" in config else tile(config)()


def add_layer_norm(x, r, w, b, eps: float) -> tuple:
    channels = x.shape[-1]
    xc, rc = x.contiguous(), r.contiguous()
    rows = xc.numel() // channels
    s, y = torch.empty_like(xc), torch.empty_like(xc)
    block = triton.next_power_of_2(channels)

    def tile(config):
        def run():
            _kernels.add_layer_norm[(triton.cdiv(rows, config["RB"]),)](
                xc,
                rc,
                w if w is not None else xc,
                b if b is not None else xc,
                s,
                y,
                rows,
                eps,
                C=channels,
                HAS_W=w is not None,
                HAS_B=b is not None,
                BC=block,
                RB=config["RB"],
                num_warps=config["num_warps"],
            )
            return s, y

        return run

    def path(name):
        fn = pytorch_path("add_layer_norm", name)
        return lambda: fn(x, r, w, b, eps)

    key = f"{x.dtype}-c{channels}-m{bucket(rows)}"
    config = choose_in("add_layer_norm", key, x.device, tile, path, NORM_TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


registry.register("linear_act", "triton", linear_act, supports)
registry.register("qkv_rope", "triton", qkv_rope, supports)
registry.register("add_layer_norm", "triton", add_layer_norm, supports)
