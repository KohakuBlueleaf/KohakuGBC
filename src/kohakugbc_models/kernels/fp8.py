"""FP8 (e4m3) linears for inference.

Recipes (``RECIPES``):

- ``row``: one fp32 scale per row of each operand (``amax / 448``): activations per token,
  weights per output channel; sm89+ (cuBLAS row-wise scaled matmul).
- ``mx``: one power-of-two e8m0 scale per 32 elements along K (``2^ceil(log2(amax / 448))``:
  the exponent rounded up, so no element overflows); sm100+ (the block-scaled MMA), scales in
  cuBLAS's 32 x 4 x 4 blocked layout.

Weights are quantized once (``quantize_weight``); activations by the kernel that produces them
(the ``quant`` ops: ``kohakugbc_models.kernels.backends.*.quant``); ``mm`` runs the GEMM through
torch's scaled matmul, the bias added in its epilogue; ``row`` on a card where cuBLAS has no
row-wise scales (sm89) through the registry's ``fp8_rowwise_mm`` (``backends.triton.fp8gemm``).
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry

E4M3_MAX = 448.0
MX_BLOCK = 32
RECIPES = ("row", "mx")
E4M3 = torch.float8_e4m3fn
_ROWWISE: dict[str, bool] = {}
_GEMMS: dict = {}


def supports(device, recipe: str) -> str | None:
    """None when ``recipe`` runs on ``device``, else why not."""
    if recipe not in RECIPES:
        return f"fp8 recipe {recipe!r}: one of {RECIPES}"
    if torch.device(device).type != "cuda":
        return "fp8 linears need CUDA"
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    if (major, minor) < (8, 9):
        return f"fp8 needs sm89+ (sm{major}{minor})"
    if recipe == "mx" and major < 10:
        return f"mx needs sm100+ (sm{major}{minor})"
    return None


def best(device) -> str | None:
    """The fastest recipe ``device`` runs (``mx``, then ``row``; measured on the RTX 5090: mx
    11% faster on SAM 3's vision encoder), None when it runs none."""
    return next((r for r in ("mx", "row") if supports(device, r) is None), None)


def quantize(x: torch.Tensor, recipe: str) -> tuple:
    """``x`` ``[M, K]`` to e4m3 ``[M, K]`` and its scales: ``row`` fp32 ``[M, 1]``, ``mx`` e8m0
    bytes ``[M, K / 32]`` (natural layout)."""
    xf = x.float()
    if recipe == "row":
        scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
        return (xf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), scale
    blocks = xf.view(x.shape[0], -1, MX_BLOCK)
    amax = blocks.abs().amax(-1).clamp(min=2.0**-126)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127)
    q = (blocks / torch.exp2(exp)[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return q.view_as(x), (exp + 127).to(torch.uint8)


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """e8m0 bytes ``[R, C]`` in cuBLAS's blocked layout (rows padded to 128, columns to 4)."""
    rows, cols = scales.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(nr * 128, nc * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    blocks = padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quantize_weight(weight: torch.Tensor, recipe: str) -> tuple:
    """A linear's weight ``[N, K]``: e4m3 and its scales in the layout ``mm`` reads."""
    q, scale = quantize(weight, recipe)
    if recipe == "row":
        return q, scale.reshape(1, -1).contiguous()
    return q, to_blocked(scale).view(torch.float8_e8m0fnu)


def cublas_rowwise(device) -> bool:
    """Whether torch's scaled matmul serves row-wise scales on ``device`` (sm89 with torch 2.14
    does not; the Triton GEMM then runs)."""
    key = str(torch.device(device))
    if key not in _ROWWISE:
        a = torch.ones(16, 16, device=device).to(E4M3)
        try:
            ones = torch.ones(16, 1, device=device)
            torch._scaled_mm(a, a.t(), ones, ones.t(), out_dtype=torch.bfloat16)
            _ROWWISE[key] = True
        except RuntimeError:
            _ROWWISE[key] = False
    return _ROWWISE[key]


def mm(xq, xs, wq, ws, recipe: str, bias=None, out_dtype=torch.bfloat16) -> torch.Tensor:
    """``x @ w.T + bias`` from e4m3 operands: ``xq`` ``[M, K]`` with its scales as the quant
    ops give them, ``wq`` / ``ws`` as ``quantize_weight`` gives them."""
    if recipe == "row":
        if cublas_rowwise(xq.device):
            return torch._scaled_mm(xq, wq.t(), xs, ws, bias=bias, out_dtype=out_dtype)
        key = str(xq.device)
        if key not in _GEMMS:
            _GEMMS[key] = registry.resolve("fp8_rowwise_mm", xq.device, E4M3)
        return _GEMMS[key](xq, xs, wq, ws, bias, out_dtype)
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        xq,
        wq.t(),
        to_blocked(xs).view(torch.float8_e8m0fnu),
        F.ScalingType.BlockWise1x32,
        ws,
        F.ScalingType.BlockWise1x32,
        swizzle_a=swizzle,
        swizzle_b=swizzle,
        bias=bias,
        output_dtype=out_dtype,
    )
