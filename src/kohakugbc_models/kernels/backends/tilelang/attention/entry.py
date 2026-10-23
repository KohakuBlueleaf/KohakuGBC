"""The TileLang attention ops and their per-problem choice: every tile config (each an nvcc
compile, cached per process) against the PyTorch paths, through ``tuning.select``; fp16
operands (Volta and Turing have no bf16 MMA)."""

from functools import cache

import torch

from kohakugbc_models.kernels import registry, tuning
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import attention as eager
from kohakugbc_models.kernels.backends.tilelang.attention._kernels import (
    LOG2E,
    attn_fwd,
    compile_kernel,
    vit_fwd,
)

TILES = [
    {"bm": bm, "bn": bn, "threads": t}
    for bm, bn, t in ((64, 64, 128), (64, 32, 128), (128, 64, 256), (128, 32, 128), (64, 64, 64))
]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]
_BOUND: dict[tuple, dict] = {}


def arch(device) -> str:
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    return f"sm_{major}{minor}"


def supports(query) -> str | None:
    if query.dtype != torch.float16:
        return f"fp16 operands only ({query.dtype})"
    return None


@cache
def vit_kernel(nwx, win, heads, d, bm, bn, threads, target, rope):
    return compile_kernel(vit_fwd(nwx, win, heads, d, "float16", bm, bn, threads, rope), target)


@cache
def attn_kernel(heads, d, has_mask, has_rpb, gh, gw, bm, bn, threads, target):
    prim = attn_fwd(heads, d, "float16", has_mask, has_rpb, gh, gw, bm, bn, threads)
    return compile_kernel(prim, target)


def choose(op: str, key: str, device, run_tile, run_torch) -> dict:
    ident = (op, key, str(device), tuning.current_metric())
    if ident not in _BOUND:
        candidates = TILES + PYTORCH

        def launch(config):
            return run_torch(config["path"]) if "path" in config else run_tile(config)

        _BOUND[ident] = tuning.select(
            f"tilelang_{op}", device, key, lambda: candidates, launch, shortlist=len(candidates)
        )
    return _BOUND[ident]


def bucket(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def pytorch_path(op: str, name: str):
    """The eager reference (``torch``) or torch.compile of it (``compile``)."""
    ref = getattr(eager, op)
    return ref if name == "torch" else compiled(op, ref)


def vit_attention(q, k, v, cos, sin, window: int, rotated: bool = False) -> torch.Tensor:
    batch, size, _, heads, dim = q.shape
    nwx = size // window
    q, k, v = (t.contiguous() for t in (q, k, v))
    cos2 = cos.reshape(window * window, dim).contiguous()
    sin2 = sin.reshape(window * window, dim).contiguous()
    out = torch.empty_like(q)
    c = dim**-0.5 * LOG2E

    def tile(config):
        kernel = vit_kernel(
            nwx,
            window,
            heads,
            dim,
            config["bm"],
            config["bn"],
            config["threads"],
            arch(q.device),
            not rotated,
        )
        return lambda: kernel(q, k, v, cos2, sin2, out, c) or out

    def path(name):
        fn = pytorch_path("vit_attention", name)
        return lambda: fn(q, k, v, cos, sin, window, rotated)

    key = f"{q.dtype}-h{heads}-d{dim}-s{size}-w{window}-b{bucket(batch)}-r{int(rotated)}"
    config = choose("vit_attention", key, q.device, tile, path)
    if "path" in config:
        return path(config["path"])()
    tile(config)()
    return out


def general(q, k, v, mask=None, rpb=None) -> torch.Tensor:
    batch, heads, lq, dim = q.shape
    lk = k.shape[2]
    qp, kp, vp = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # physical [B, L, H, D]
    out = torch.empty_like(qp)
    if mask is not None:
        key_mask = mask.reshape(batch, -1, mask.shape[-1])[:, 0].to(torch.int8).contiguous()
    else:
        key_mask = torch.ones(batch, lk, dtype=torch.int8, device=q.device)
    if rpb is not None:
        y, x = (t.contiguous() for t in rpb)
    else:
        y = x = torch.zeros(batch, heads, lq, 1, dtype=q.dtype, device=q.device)
    c = dim**-0.5 * LOG2E

    def tile(config):
        kernel = attn_kernel(
            heads,
            dim,
            mask is not None,
            rpb is not None,
            y.shape[-1],
            x.shape[-1],
            config["bm"],
            config["bn"],
            config["threads"],
            arch(q.device),
        )
        return lambda: kernel(qp, kp, vp, key_mask, y, x, out, c) or out

    def path(name):
        if rpb is not None:
            fn = pytorch_path("attention_rpb", name)
            return lambda: fn(q, k, v, *rpb)
        fn = pytorch_path("attention", name)
        return lambda: fn(q, k, v, mask)

    op = "attention_rpb" if rpb is not None else "attention"
    key = (
        f"{q.dtype}-h{heads}-d{dim}-q{bucket(lq)}-k{bucket(lk)}-m{int(mask is not None)}"
        f"-b{bucket(batch)}"
    )
    config = choose(op, key, q.device, tile, path)
    if "path" in config:
        return path(config["path"])()
    tile(config)()
    return out.transpose(1, 2)


def attention(q, k, v, mask=None) -> torch.Tensor:
    return general(q, k, v, mask=mask)


def attention_rpb(q, k, v, bias_y, bias_x) -> torch.Tensor:
    return general(q, k, v, rpb=(bias_y, bias_x))


registry.register("vit_attention", "tilelang", vit_attention, supports)
registry.register("attention", "tilelang", attention, supports)
registry.register("attention_rpb", "tilelang", attention_rpb, supports)
