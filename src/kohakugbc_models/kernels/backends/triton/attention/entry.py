"""The Triton attention ops and their per-problem choice.

Each call's problem (op, dtype, heads, head dim, lengths, window, batch bucket) picks, through
``tuning.select``, among every Triton tile config (BM, BN, warps, stages) and the PyTorch paths
(``torch``: the eager reference; ``compile``: torch.compile of it), so a kernel binds only where
it is measured faster. The choice is bound per problem key, so a call looks it up once.
"""

import torch
import triton

from kohakugbc_models.kernels import registry, tuning
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import attention as eager
from kohakugbc_models.kernels.backends.triton.attention._kernels import attn_fwd, vit_fwd, vit_rope
from kohakugbc_models.kernels.hardware import tiles_for

DTYPES = (torch.bfloat16, torch.float16)
TILES = [
    {"BM": bm, "BN": bn, "num_warps": w, "num_stages": s}
    for bm in (64, 128)
    for bn in (32, 64, 128)
    for w in (4, 8)
    for s in (2, 3)
    if not (bm == 128 and w == 4 and bn == 128)
]
# sm10x: larger query / key blocks and deeper pipelines (228 KiB of shared memory per SM)
EXTRA = {
    "sm10x": [
        {"BM": bm, "BN": bn, "num_warps": w, "num_stages": s}
        for bm, bn, w, s in (
            (128, 128, 8, 4),
            (128, 256, 8, 2),
            (256, 64, 8, 3),
            (256, 128, 16, 2),
        )
    ]
}
PYTORCH = [{"path": "torch"}, {"path": "compile"}]
_BOUND: dict[tuple, object] = {}


def bucket(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def pytorch_path(op: str, name: str):
    """The eager reference (``torch``) or torch.compile of it (``compile``)."""
    ref = getattr(eager, op)
    return ref if name == "torch" else compiled(op, ref)


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def choose(op: str, key: str, device, run_tile, run_torch, tiles=None) -> object:
    """The bound config of ``(op, key)``: a tile config or a PyTorch path."""
    tiles = tiles_for(device, TILES, EXTRA) if tiles is None else tiles
    return choose_in(op, key, device, run_tile, run_torch, tiles + PYTORCH)


def choose_in(op: str, key: str, device, run_tile, run_torch, candidates: list) -> object:
    """The bound config of ``(op, key)`` among ``candidates`` (``{"path": name}`` ones run
    ``run_torch(name)``, the rest ``run_tile(config)``)."""
    ident = (op, key, str(device), tuning.current_metric())
    if ident not in _BOUND:

        def launch(config):
            if "path" in config:
                return run_torch(config["path"])
            return run_tile(config)

        config = tuning.select(
            f"triton_{op}", device, key, lambda: candidates, launch, shortlist=len(candidates)
        )
        _BOUND[ident] = config
    return _BOUND[ident]


# ``fused``: q / k rotated inside the attention kernel (every query block re-rotates every key
# tile); ``pass``: rotated once by ``vit_rope``, then attention without RoPE
VIT_TILES = [{**t, "rope": r} for t in TILES for r in ("fused", "pass")]
VIT_EXTRA = {
    f: [{**t, "rope": r} for t in ts for r in ("fused", "pass")] for f, ts in EXTRA.items()
}
ROPE_ROWS = 32


def vit_attention(q, k, v, cos, sin, window: int, rotated: bool = False) -> torch.Tensor:
    batch, size, _, heads, dim = q.shape
    nwx = size // window
    if not rotated:
        q, k = q.contiguous(), k.contiguous()
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    length = window * window
    cos2, sin2 = cos.reshape(length, dim), sin.reshape(length, dim)

    def tile(config):
        grid = (triton.cdiv(length, config["BM"]), batch * nwx * nwx * heads)
        rows = batch * size * size * heads

        def run():
            qa, ka, fused = q, k, config["rope"] == "fused" and not rotated
            if not fused and not rotated:
                qa, ka = torch.empty_like(q), torch.empty_like(k)
                vit_rope[(triton.cdiv(rows, ROPE_ROWS),)](
                    q,
                    k,
                    qa,
                    ka,
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
            vit_fwd[grid](
                qa,
                ka,
                v,
                out,
                cos2,
                sin2,
                *qa.stride()[:4],
                *ka.stride()[:4],
                *v.stride()[:4],
                *out.stride()[:4],
                dim**-0.5,
                NWX=nwx,
                WIN=window,
                HEADS=heads,
                D=dim,
                BM=config["BM"],
                BN=config["BN"],
                ROPE=fused,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            return out

        return run

    def path(name):
        fn = pytorch_path("vit_attention", name)
        return lambda: fn(q, k, v, cos, sin, window, rotated)

    # Rotated operands have no RoPE choice to make: only the tiles' first RoPE variant
    tiles = tiles_for(q.device, VIT_TILES, VIT_EXTRA)
    tiles = [t for t in tiles if t["rope"] == "pass"] if rotated else tiles
    key = f"{q.dtype}-h{heads}-d{dim}-s{size}-w{window}-b{bucket(batch)}-r{int(rotated)}"
    config = choose("vit_attention", key, q.device, tile, path, tiles)
    return path(config["path"])() if "path" in config else tile(config)()


def general(q, k, v, mask=None, rpb=None) -> torch.Tensor:
    batch, heads, lq, dim = q.shape
    lk = k.shape[2]
    out = torch.empty(batch, lq, heads, dim, dtype=q.dtype, device=q.device).transpose(1, 2)
    key_mask = None
    if mask is not None:
        key_mask = mask.reshape(batch, -1, mask.shape[-1])[:, 0].to(torch.int8).contiguous()
    y, x = rpb if rpb is not None else (None, None)

    def tile(config):
        grid = (triton.cdiv(lq, config["BM"]), batch * heads)
        dummy = q

        def run():
            m, ty, tx = key_mask, y, x
            attn_fwd[grid](
                q,
                k,
                v,
                out,
                m if m is not None else dummy,
                ty if ty is not None else dummy,
                tx if tx is not None else dummy,
                *q.stride()[:3],
                *k.stride()[:3],
                *v.stride()[:3],
                *out.stride()[:3],
                m.stride(0) if m is not None else 0,
                *(ty.stride() if ty is not None else (0, 0, 0, 0)),
                *(tx.stride() if tx is not None else (0, 0, 0, 0)),
                lq,
                lk,
                dim**-0.5,
                HEADS=heads,
                D=dim,
                GW=tx.shape[-1] if tx is not None else 1,
                HAS_MASK=m is not None,
                HAS_RPB=ty is not None,
                BM=config["BM"],
                BN=config["BN"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            return out

        return run

    def path(name):
        if rpb is not None:
            fn = pytorch_path("attention_rpb", name)
            return lambda: fn(q, k, v, y, x)
        fn = pytorch_path("attention", name)
        return lambda: fn(q, k, v, mask)

    op = "attention_rpb" if rpb is not None else "attention"
    key = (
        f"{q.dtype}-h{heads}-d{dim}-q{bucket(lq)}-k{bucket(lk)}-m{int(mask is not None)}"
        f"-b{bucket(batch)}"
    )
    config = choose(op, key, q.device, tile, path)
    return path(config["path"])() if "path" in config else tile(config)()


def attention(q, k, v, mask=None) -> torch.Tensor:
    return general(q, k, v, mask=mask)


def attention_rpb(q, k, v, bias_y, bias_x) -> torch.Tensor:
    return general(q, k, v, rpb=(bias_y, bias_x))


registry.register("vit_attention", "triton", vit_attention, supports)
registry.register("attention", "triton", attention, supports)
registry.register("attention_rpb", "triton", attention_rpb, supports)
