"""Shared parts of the persistent, warp-specialized Gluon GEMMs (sm10x: tcgen05 + TMEM): the
grouped tile order, the TMA loader, the MMA issuer over a STAGES-deep shared-memory ring into two
TMEM accumulators, barrier setup and the staged TMA store of the epilogues.

Partitions of every kernel: the epilogue (default partition), 1 TMA load warp, 1 MMA warp.
``acc_ready`` / ``acc_free`` hand each accumulator between the MMA warp and the epilogue, so one
tile's epilogue overlaps the next tile's main loop.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared


@gluon.jit
def tile_of(t, num_m, num_n, GROUP_M: gl.constexpr):
    """Tile ``t``'s (row, column) block, walked in ``GROUP_M``-row bands (a band's W tiles stay
    in L2)."""
    group = GROUP_M * num_n
    first_m = (t // group) * GROUP_M
    size_m = gl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (t % group) % size_m
    pid_n = (t % group) // size_m
    return pid_m, pid_n


@gluon.jit
def load_nt(
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
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    """A ``[M, K]`` tiles and B ``[N, K]`` tiles (``A @ B^T``, the nn.Linear layout)."""
    count = 0
    nbytes: gl.constexpr = a_desc.block_type.nbytes + b_desc.block_type.nbytes
    for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        pid_m, pid_n = tile_of(t, num_m, num_n, GROUP_M)
        for kk in range(k_tiles):
            s = count % STAGES
            mbarrier.wait(free.index(s), ((count // STAGES) & 1) ^ 1)
            mbarrier.expect(ready.index(s), nbytes)
            tma.async_copy_global_to_shared(
                a_desc, [pid_m * BM, kk * BK], ready.index(s), a_smem.index(s)
            )
            tma.async_copy_global_to_shared(
                b_desc, [pid_n * BN, kk * BK], ready.index(s), b_smem.index(s)
            )
            count += 1


@gluon.jit
def mma(
    a_smem,
    b_smem,
    acc,
    ready,
    free,
    acc_ready,
    acc_free,
    num_tiles,
    k_tiles,
    STAGES: gl.constexpr,
):
    count = 0
    local = 0
    for _t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        buf = local % 2
        mbarrier.wait(acc_free.index(buf), ((local // 2) & 1) ^ 1)
        acc_buf = acc.index(buf)
        for kk in range(k_tiles):
            s = count % STAGES
            mbarrier.wait(ready.index(s), (count // STAGES) & 1)
            tcgen05_mma(
                a_smem.index(s),
                b_smem.index(s).permute((1, 0)),
                acc_buf,
                use_acc=kk > 0,
                mbarriers=[free.index(s)],
            )
            count += 1
        tcgen05_commit(acc_ready.index(buf))
        local += 1  # noqa: SIM113 (Gluon has no enumerate)


@gluon.jit
def setup(STAGES: gl.constexpr):
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    free = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    fence_async_shared()
    return ready, free, acc_ready, acc_free


@gluon.jit
def put(desc, row, col, vals, c_smem, stores):
    """TMA-store ``vals`` at ``[row, col]`` through the next of ``c_smem``'s staging slots, once
    that slot's previous store has landed (up to SLOTS - 1 in flight)."""
    SLOTS: gl.constexpr = c_smem.shape[0]
    slot = stores % SLOTS
    tma.store_wait(pendings=SLOTS - 1)
    c_smem.index(slot).store(vals)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc, [row, col], c_smem.index(slot))
    return stores + 1
