"""The Gluon GEMM (sm10x): ``X [M, K] @ W [N, K].T + bias`` with the Triton ``gemm``'s
epilogues (``backends.triton.blocks._kernels``), each value rounded where the eager path rounds
it: the linear rounded to the output dtype, then in fp32 ``ACT`` 0 none / 1 GELU (``_gelu``) /
2 ReLU;
``ROPE``: the ViT's fused q / k / v projection (``N = 3C``, ``C`` a multiple of ``BN``), the
columns of q and k rotated pairwise by the row's in-window position (row ``m`` is grid cell
``m % SIZE^2``), q / k / v stored through their own descriptors (``[M, C]`` each).

Persistent and warp-specialized (``_common``): a TMA load warp and an MMA warp fill two TMEM
accumulators of ``[BM, BN]``; the epilogue drains one in ``[BM, EPI]`` chunks through
``SLOTS`` staging buffers while the next tile's main loop runs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    mbarrier,
    tma,
)

from kohakugbc_models.kernels.backends.gluon.gemm._common import (
    load_nt,
    mma,
    put,
    setup,
    tile_of,
)


@gluon.jit
def _gelu(y):
    """GELU as ``y / 2 + |y| (1 - e) / 2``, ``e = 1 - erf(|y| / sqrt 2)`` by Abramowitz-Stegun
    7.1.26 with the hardware reciprocal and exp2 (absolute error under 5e-7, below a bf16 step;
    libdevice's erf made the epilogue ALU-bound)."""
    z = gl.abs(y) * 0.7071067811865476
    t = gl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [1.0 + 0.3275911 * z],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    ex = gl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=r,r",
        [-z * z * 1.4426950408889634],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return 0.5 * y + 0.5 * gl.abs(y) * (1.0 - poly * ex)


@gluon.jit
def _rope(
    v,
    row,
    col,
    cos_ptr,
    sin_ptr,
    M,
    SIZE,
    WIN,
    D,
    BM: gl.constexpr,
    EPI: gl.constexpr,
    regs: gl.constexpr,
):
    """``v`` ``[BM, EPI]`` (rows from ``row``, columns from ``col``) rotated pairwise."""
    half: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [BM, EPI // 2], TensorMemoryLayout([BM, EPI // 2], col_stride=1), gl.num_warps()
    )
    ye, yo = gl.split(gl.reshape(v, [BM, EPI // 2, 2]))
    ye = gl.convert_layout(ye, half)
    yo = gl.convert_layout(yo, half)
    rows = row + gl.arange(0, BM, layout=gl.SliceLayout(1, half))
    cell = rows % (SIZE * SIZE)
    t = ((cell // SIZE) % WIN) * WIN + (cell % SIZE) % WIN
    even = (col + 2 * gl.arange(0, EPI // 2, layout=gl.SliceLayout(0, half))) % D
    table = t[:, None] * D + even[None, :]
    keep = (rows < M)[:, None]
    ce = gl.load(cos_ptr + table, mask=keep, other=0.0)
    se = gl.load(sin_ptr + table, mask=keep, other=0.0)
    co = gl.load(cos_ptr + table + 1, mask=keep, other=0.0)
    so = gl.load(sin_ptr + table + 1, mask=keep, other=0.0)
    out = gl.join(ye * ce - yo * se, yo * co + ye * so)
    return gl.convert_layout(gl.reshape(out, [BM, EPI]), regs)


@gluon.jit
def _epilogue(
    q_desc,
    k_desc,
    v_desc,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    acc,
    c_smem,
    acc_ready,
    acc_free,
    num_tiles,
    num_m,
    num_n,
    M,
    N,
    C,
    SIZE,
    WIN,
    D,
    BM: gl.constexpr,
    BN: gl.constexpr,
    EPI: gl.constexpr,
    GROUP_M: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    ACT: gl.constexpr,
    ROPE: gl.constexpr,
):
    chunk: gl.constexpr = TensorMemoryLayout([BM, EPI], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [BM, EPI], chunk, gl.num_warps())
    dtype: gl.constexpr = q_desc.dtype
    local = 0
    stores = 0
    for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        pid_m, pid_n = tile_of(t, num_m, num_n, GROUP_M)
        buf = local % 2
        mbarrier.wait(acc_ready.index(buf), (local // 2) & 1)
        acc_buf = acc.index(buf)
        row = pid_m * BM
        which = (pid_n * BN) // C
        base = pid_n * BN - which * C
        for c in gl.static_range(BN // EPI):
            v = acc_buf.slice(c * EPI, EPI).load(regs)
            if c == BN // EPI - 1:
                mbarrier.arrive(acc_free.index(buf), count=1)
            col = pid_n * BN + c * EPI
            if HAS_BIAS:
                cols = col + gl.arange(0, EPI, layout=gl.SliceLayout(0, regs))
                v = v + gl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(gl.float32)[None, :]
            if ACT != 0 or ROPE:
                v = v.to(dtype).to(gl.float32)
            if ACT == 1:
                v = _gelu(v)
            elif ACT == 2:
                v = gl.maximum(v, 0.0)
            out_col = base + c * EPI
            if ROPE:
                if which == 2:
                    stores = put(v_desc, row, out_col, v.to(dtype), c_smem, stores)
                else:
                    v = _rope(v, row, out_col, cos_ptr, sin_ptr, M, SIZE, WIN, D, BM, EPI, regs)
                    if which == 0:
                        stores = put(q_desc, row, out_col, v.to(dtype), c_smem, stores)
                    else:
                        stores = put(k_desc, row, out_col, v.to(dtype), c_smem, stores)
            else:
                stores = put(q_desc, row, out_col, v.to(dtype), c_smem, stores)
        local += 1  # noqa: SIM113 (Gluon has no enumerate)
    tma.store_wait(pendings=0)


@gluon.jit
def gemm(
    a_desc,
    b_desc,
    q_desc,
    k_desc,
    v_desc,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    M,
    N,
    K,
    C,
    SIZE,
    WIN,
    D,
    HAS_BIAS: gl.constexpr,
    ACT: gl.constexpr,
    ROPE: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    EPI: gl.constexpr,
    STAGES: gl.constexpr,
    SLOTS: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    num_m = gl.cdiv(M, BM)
    num_n = gl.cdiv(N, BN)
    num_tiles = num_m * num_n
    k_tiles = gl.cdiv(K, BK)
    dtype: gl.constexpr = a_desc.dtype
    a_smem = gl.allocate_shared_memory(dtype, [STAGES, BM, BK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [STAGES, BN, BK], b_desc.layout)
    c_smem = gl.allocate_shared_memory(dtype, [SLOTS, BM, EPI], q_desc.layout)
    ready, free, acc_ready, acc_free = setup(STAGES)
    acc = allocate_tensor_memory(
        gl.float32, [2, BM, BN], TensorMemoryLayout([BM, BN], col_stride=1)
    )
    gl.warp_specialize(
        [
            (
                _epilogue,
                (
                    q_desc,
                    k_desc,
                    v_desc,
                    bias_ptr,
                    cos_ptr,
                    sin_ptr,
                    acc,
                    c_smem,
                    acc_ready,
                    acc_free,
                    num_tiles,
                    num_m,
                    num_n,
                    M,
                    N,
                    C,
                    SIZE,
                    WIN,
                    D,
                    BM,
                    BN,
                    EPI,
                    GROUP_M,
                    HAS_BIAS,
                    ACT,
                    ROPE,
                ),
            ),
            (
                load_nt,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    ready,
                    free,
                    num_tiles,
                    num_m,
                    num_n,
                    k_tiles,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    GROUP_M,
                ),
            ),
            (
                mma,
                (a_smem, b_smem, acc, ready, free, acc_ready, acc_free, num_tiles, k_tiles, STAGES),
            ),
        ],
        [1, 1],
        [24, 24],
    )
