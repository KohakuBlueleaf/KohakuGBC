"""The Gluon GEMM ops (sm10x) and their per-problem choice: ``linear_act`` and ``qkv_rope`` as
the Triton block ops compute them (``backends.triton.blocks``), on a persistent tcgen05 GEMM.

Each problem picks, through ``tuning.select`` (tables ``gluon_<op>``), among the Gluon configs
and the paths ``triton`` (the Triton entry, itself tuned), for a plain linear ``cublas``
(``F.linear``) and for ``qkv_rope`` ``split`` (one cuBLAS GEMM, then ``vit_rope`` rotating q and
k in place). A problem the GEMM cannot address (a row stride TMA cannot take, a ``C`` the
column tile does not divide) runs the Triton entry.
"""

import torch
import torch.nn.functional as F
import triton
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from kohakugbc_models.kernels import registry, tuning
from kohakugbc_models.kernels.backends.gluon.gemm import _kernels
from kohakugbc_models.kernels.backends.triton.attention._kernels import vit_rope
from kohakugbc_models.kernels.backends.triton.attention.entry import ROPE_ROWS, bucket
from kohakugbc_models.kernels.backends.triton.blocks import entry as triton_blocks

DTYPES = {torch.bfloat16: gl.bfloat16, torch.float16: gl.float16}
ACT = {"none": 0, "gelu": 1, "relu": 2}
# (BM, BN, BK, EPI, STAGES, SLOTS, warps of the epilogue): 228 KiB of shared memory per SM
CONFIGS = [
    {"BM": 128, "BN": bn, "BK": bk, "EPI": epi, "STAGES": st, "SLOTS": sl, "num_warps": w}
    for bn, bk, epi, st, sl, w in (
        (256, 64, 64, 4, 2, 4),
        (256, 64, 64, 3, 3, 4),
        (256, 64, 64, 4, 2, 8),
        (256, 64, 32, 4, 3, 4),
        (128, 64, 64, 6, 2, 4),
        (128, 128, 64, 3, 2, 4),
    )
]
_SMS: dict[int, int] = {}
_BOUND: dict[tuple, dict] = {}


def supports(query) -> str | None:
    if query.dtype not in DTYPES:
        return f"16-bit operands only ({query.dtype})"
    return None


def sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def desc(t: torch.Tensor, block: list[int]) -> TensorDescriptor:
    layout = gl.NVMMASharedLayout.get_default_for(block, DTYPES[t.dtype])
    return TensorDescriptor.from_tensor(t, block, layout)


def addressable(*tensors: torch.Tensor) -> bool:
    """Every 2D operand TMA can address: unit inner stride, 16-byte aligned rows and base."""
    return all(
        t.stride(-1) == 1 and (t.stride(0) * t.element_size()) % 16 == 0 and t.data_ptr() % 16 == 0
        for t in tensors
    )


def launch(x2, w, b, outs, cos, sin, config, act=0, rope=None) -> None:
    """``outs``: the one output ``[M, N]``, or q / k / v ``[M, C]`` each with ``rope = (size,
    window, dim, C)``."""
    m, k = x2.shape
    n = w.shape[0]
    size, window, dim, channels = rope if rope is not None else (1, 1, 2, n)
    bm, bn, bk, epi = config["BM"], config["BN"], config["BK"], config["EPI"]
    out_descs = [desc(o, [bm, epi]) for o in outs]
    out_descs += [out_descs[0]] * (3 - len(out_descs))
    tiles = -(-m // bm) * -(-n // bn)
    _kernels.gemm[(min(tiles, sms(x2.device)),)](
        desc(x2, [bm, bk]),
        desc(w, [bn, bk]),
        *out_descs,
        b if b is not None else w,
        cos,
        sin,
        m,
        n,
        k,
        channels,
        size,
        window,
        dim,
        HAS_BIAS=b is not None,
        ACT=act,
        ROPE=rope is not None,
        BM=bm,
        BN=bn,
        BK=bk,
        EPI=epi,
        STAGES=config["STAGES"],
        SLOTS=config["SLOTS"],
        GROUP_M=8,
        num_warps=config["num_warps"],
    )


def choose(op: str, key: str, device, run_config, run_path, candidates: list) -> dict:
    ident = (op, key, str(device), tuning.current_metric())
    if ident not in _BOUND:

        def runner(config):
            return run_path(config["path"]) if "path" in config else run_config(config)

        _BOUND[ident] = tuning.select(
            f"gluon_{op}", device, key, lambda: candidates, runner, shortlist=len(candidates)
        )
    return _BOUND[ident]


def linear_act(x, w, b, act: str = "none") -> torch.Tensor:
    k, n = x.shape[-1], w.shape[0]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if not addressable(x2, w) or n % 8:
        return triton_blocks.linear_act(x, w, b, act)
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)

    def run_config(config):
        def run():
            launch(x2, w, b, [out], w, w, config, ACT[act])
            return out.view(*x.shape[:-1], n)

        return run

    def run_path(name):
        if name == "cublas":
            return lambda: F.linear(x, w, b)
        return lambda: triton_blocks.linear_act(x, w, b, act)

    paths = [{"path": "triton"}] + ([{"path": "cublas"}] if act == "none" else [])
    key = f"{x.dtype}-m{bucket(m)}-n{n}-k{k}-{act}-b{int(b is not None)}"
    config = choose("linear_act", key, x.device, run_config, run_path, CONFIGS + paths)
    return run_path(config["path"])() if "path" in config else run_config(config)()


def qkv_rope(x, w, b, cos, sin, window: int) -> tuple:
    batch, size, _, channels = x.shape
    dim = cos.shape[-1]
    heads = channels // dim
    length = window * window
    cos2 = cos.reshape(length, dim).contiguous()
    sin2 = sin.reshape(length, dim).contiguous()
    x2 = x.reshape(-1, channels)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    shape = (batch, size, size, heads, dim)
    fits = [c for c in CONFIGS if channels % c["BN"] == 0 and c["EPI"] % 2 == 0]
    if not addressable(x2, w) or not fits or dim % 2 or channels % 8:
        return triton_blocks.qkv_rope(x, w, b, cos, sin, window)

    def run_config(config):
        out = torch.empty(3, m, channels, dtype=x.dtype, device=x.device)

        def run():
            launch(x2, w, b, list(out), cos2, sin2, config, rope=(size, window, dim, channels))
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

    def run_path(name):
        return (
            split if name == "split" else lambda: triton_blocks.qkv_rope(x, w, b, cos, sin, window)
        )

    paths = [{"path": "split"}, {"path": "triton"}]
    key = f"{x.dtype}-c{channels}-d{dim}-s{size}-w{window}-b{bucket(batch)}"
    config = choose("qkv_rope", key, x.device, run_config, run_path, fits + paths)
    return run_path(config["path"])() if "path" in config else run_config(config)()


registry.register("linear_act", "gluon", linear_act, supports)
registry.register("qkv_rope", "gluon", qkv_rope, supports)
