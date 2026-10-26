"""Low-precision linears for inference: FP8 (e4m3) and NVFP4.

Recipes (``RECIPES``):

- ``row``: one fp32 scale per row of each operand (``amax / 448``): activations per token,
  weights per output channel; sm89+ (cuBLAS row-wise scaled matmul).
- ``mx``: one power-of-two e8m0 scale per 32 elements along K (``2^ceil(log2(amax / 448))``:
  the exponent rounded up, so no element overflows); sm100+ (the block-scaled MMA), scales in
  cuBLAS's 32 x 4 x 4 blocked layout.
- ``nvfp4``: E2M1 values packed two per byte (the even element low), one e4m3 scale per 16
  along K (``gs * amax / 6``, round to nearest) and an fp32 tensor scale ``gs = 448 * 6 /
  amax`` per operand; sm100+, scales blocked as ``mx``'s. Activation scales travel as
  ``(blocked e4m3, 1 / gs)``. An activation's ``amax`` is its own, taken each call (an extra
  read of it), or a calibrated constant (``Fp4Input``: one per linear input).

Weights are quantized once (``quantize_weight``); activations by the kernel that produces them
(the ``quant`` ops: ``kohakugbc_models.kernels.backends.*.quant``); ``mm`` runs the GEMM through
torch's scaled matmul, the bias added in its epilogue; ``row`` on a card where cuBLAS has no
row-wise scales (sm89) through the registry's ``fp8_rowwise_mm`` (``backends.triton.fp8gemm``).
"""

import torch
import torch.nn.functional as F
from torch import nn

from kohakugbc_models.kernels import registry

E4M3_MAX = 448.0
E2M1_MAX = 6.0
MX_BLOCK = 32
FP4_BLOCK = 16
RECIPES = ("row", "mx", "nvfp4")
E4M3 = torch.float8_e4m3fn
# E2M1's magnitudes in code order and the rounding boundaries between them; a tie goes to the
# even code, so the boundary belongs to the code below (0.25 -> 0) or above (0.75 -> 1.0)
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_UP = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True))
E2M1_UP += ((5.0, False),)
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
    if recipe in ("mx", "nvfp4") and major < 10:
        return f"{recipe} needs sm100+ (sm{major}{minor})"
    return None


def best(device) -> str | None:
    """The fastest recipe ``device`` runs (``mx``, then ``row``; measured on the RTX 5090: mx
    11% faster on SAM 3's vision encoder), None when it runs none."""
    return next((r for r in ("mx", "row") if supports(device, r) is None), None)


class Fp4Input(nn.Module):
    """One linear's NVFP4 input scale. ``g`` holds ``(amax, 1 / gs)``: with ``static`` the
    producers read the calibrated ``amax`` (one pass, values past it saturate); else they take
    the activation's own each call into a scratch ``g`` and, with ``record``, fold it into
    ``seen`` (the calibration's running max)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("g", torch.zeros(2))
        self.register_buffer("seen", torch.zeros(1))
        self.static = self.record = False

    def set(self, amax: float) -> None:
        self.g.copy_(torch.tensor([amax, amax / (E4M3_MAX * E2M1_MAX)]))
        self.static = True

    def gs(self) -> torch.Tensor:
        return (E4M3_MAX * E2M1_MAX / self.g[:1].clamp(min=1e-12)).float()


def fp4_inputs(module: nn.Module) -> dict[str, Fp4Input]:
    return {n: m for n, m in module.named_modules() if isinstance(m, Fp4Input)}


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes of fp32 ``v``: round to nearest even, saturate at 6, the sign bit kept
    for a negative value that rounds to zero (as the hardware conversion)."""
    a = v.abs()
    code = torch.zeros_like(a, dtype=torch.int32)
    for edge, inclusive in E2M1_UP:
        code += (a >= edge if inclusive else a > edge).to(torch.int32)
    return (code | ((v < 0).to(torch.int32) << 3)).to(torch.uint8)


def quantize_fp4(x: torch.Tensor, gs: torch.Tensor | None = None) -> tuple:
    """``x`` ``[M, K]`` to packed E2M1 ``[M, K / 2]``, e4m3 block scales ``[M, K / 16]``
    (natural layout) and the tensor scale ``gs`` (fp32 ``[1]``; from ``x``'s amax if None)."""
    xf = x.float()
    if gs is None:
        gs = (E4M3_MAX * E2M1_MAX / xf.abs().amax().clamp(min=1e-12)).reshape(1)
    rows, k = xf.shape
    blocks = xf.view(rows, k // FP4_BLOCK, FP4_BLOCK)
    sf = (gs * blocks.abs().amax(-1) / E2M1_MAX).clamp(max=E4M3_MAX).to(E4M3)
    step = sf.float() / gs
    codes = e2m1_codes(torch.where(step[..., None] > 0, blocks / step[..., None], 0.0))
    codes = codes.view(rows, k)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), sf, gs


def quantize(x: torch.Tensor, recipe: str, site: Fp4Input | None = None) -> tuple:
    """``x`` ``[M, K]`` to e4m3 ``[M, K]`` and its scales: ``row`` fp32 ``[M, 1]``, ``mx`` e8m0
    bytes ``[M, K / 32]`` (natural layout); ``nvfp4``: packed E2M1 ``[M, K / 2]`` and ``(blocked
    e4m3 scales, 1 / gs)``, ``gs`` from ``site`` when it is static."""
    if recipe == "nvfp4":
        static = site is not None and site.static
        q, sf, gs = quantize_fp4(x, site.gs() if static else None)
        if site is not None and site.record and not static:
            torch.maximum(site.seen, E4M3_MAX * E2M1_MAX / gs, out=site.seen)
        return q, (to_blocked(sf.view(torch.uint8)).view(E4M3), 1.0 / gs)
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
    """Scale bytes ``[R, C]`` (e8m0 or e4m3) in cuBLAS's blocked layout (rows padded to 128,
    columns to 4)."""
    rows, cols = scales.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(nr * 128, nc * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    blocks = padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quantize_weight(weight: torch.Tensor, recipe: str) -> tuple:
    """A linear's weight ``[N, K]``: e4m3 and its scales in the layout ``mm`` reads."""
    q, scale = quantize(weight, recipe)
    if recipe == "nvfp4":
        return q, scale
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
    if recipe == "nvfp4":
        fp4 = torch.float4_e2m1fn_x2
        tensor_wise = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        return F.scaled_mm(
            xq.view(fp4),
            wq.view(fp4).t(),
            [xs[0].flatten(), xs[1]],
            tensor_wise,
            [ws[0].flatten(), ws[1]],
            tensor_wise,
            swizzle_a=[swizzle, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[swizzle, F.SwizzleType.NO_SWIZZLE],
            bias=bias,
            output_dtype=out_dtype,
        )
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
