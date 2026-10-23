"""Triton fp8 GEMM with row-wise scales, for cards whose cuBLAS has no row-wise scaled matmul
(sm89 with torch 2.14): ``C = (A . B^T) * sa[m] * sb[n] + bias``, ``A`` ``[M, K]`` and ``B``
``[N, K]`` e4m3, fp32 scales, 16-bit ``C``. Each ``BK`` slice's product is taken into a fresh
fp32 accumulator and added in fp32 (Ada's fp8 MMA accumulates with fewer bits).
"""

import triton
import triton.language as tl


@triton.jit
def rowwise_mm(
    A,
    B,
    SA,
    SB,
    BIAS,
    C,
    M,
    N,
    K,
    sam,
    sbn,
    scm,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    band = GROUP * grid_n
    first = (pid // band) * GROUP
    rows = tl.minimum(grid_m - first, GROUP)
    pid_m = first + (pid % band) % rows
    pid_n = (pid % band) // rows
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    in_m = offs_m < M
    in_n = offs_n < N
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :]
    b_ptr = B + offs_n[None, :] * sbn + offs_k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        in_k = offs_k < K - k * BK
        a = tl.load(a_ptr, mask=in_m[:, None] & in_k[None, :], other=0.0)
        b = tl.load(b_ptr, mask=in_n[None, :] & in_k[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BK
        b_ptr += BK
    sa = tl.load(SA + offs_m, mask=in_m, other=0.0)
    sb = tl.load(SB + offs_n, mask=in_n, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=in_n, other=0.0).to(tl.float32)[None, :]
    c_ptr = C + offs_m[:, None] * scm + offs_n[None, :]
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=in_m[:, None] & in_n[None, :])
