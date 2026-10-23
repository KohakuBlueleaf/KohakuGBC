"""The block kernels' exact properties, on every GPU backend that serves them here:

- rows are independent: the op on a row slice equals the slice of the op, bit for bit;
- ``add_layer_norm``'s sum is ``x + r`` exactly (one fp32 add, one rounding, as torch's);
- ``qkv_rope``'s v (no rotary) equals ``linear_act`` over v's weight rows, bit for bit;
- a broken variant (another activation, rotary tables of the wrong window grid, swapped norm
  affine) lands far from fp64 next to the correct call (precision itself is reviewed in the
  bench's profiles, not judged here).
"""

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.detect.sam3.vision import rope_tables  # noqa: E402
from kohakugbc_models.kernels import registry, tuning  # noqa: E402
from kohakugbc_models.kernels.backends.eager import blocks as eager  # noqa: E402

ENTRIES = []
try:
    from kohakugbc_models.kernels.backends.triton.attention import entry as triton_attention

    ENTRIES.append(triton_attention)
except ImportError:
    pass

pytestmark = pytest.mark.gpu
CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def one_tile(tmp_path, monkeypatch):
    """Every problem binds its first config, untimed (the same config at every row count)."""
    monkeypatch.setenv("KOHAKUGBC_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuning, "_CHOSEN", {})
    monkeypatch.setattr(tuning, "_TABLES", {})
    registry.load_backends()
    for module in ENTRIES:
        monkeypatch.setattr(module, "_BOUND", {})
    tuning.set_mode("off")
    yield
    tuning.set_mode("table")


def served(op: str, backend: str) -> torch.dtype:
    if not CUDA:
        pytest.skip("no CUDA device")
    dtype = torch.float16 if torch.cuda.get_device_capability(0)[0] < 8 else torch.bfloat16
    if registry.available(op, "cuda:0", dtype)[backend] is not None:
        pytest.skip(f"{backend} does not serve {op} here")
    return dtype


def rnd(*shape, dtype, scale=1.0):
    return (torch.randn(*shape, device="cuda:0") * scale).to(dtype)


def far(err_broken: float, err_ok: float) -> bool:
    return err_broken > 100 * max(err_ok, 1e-30)


def mean_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().mean().item()


@pytest.mark.parametrize("backend", ["triton"])
@pytest.mark.parametrize("act", ["gelu", "relu"])
def test_linear_act_rows(backend, act):
    dtype = served("linear_act", backend)
    op = registry.resolve("linear_act", "cuda:0", dtype, backend=backend)
    torch.manual_seed(0)
    x = rnd(3, 300, 256, dtype=dtype)
    w, b = rnd(1000, 256, dtype=dtype, scale=0.06), rnd(1000, dtype=dtype)
    with torch.inference_mode():
        full = op(x, w, b, act)
        assert torch.equal(full[1:2], op(x[1:2], w, b, act))
        assert torch.equal(full[:, 7:180], op(x[:, 7:180].contiguous(), w, b, act))
        ruler = eager.linear_act(x.double(), w.double(), b.double(), act)
        broken = op(x, w, b, "none")
    assert far(mean_err(broken, ruler), mean_err(full, ruler))


@pytest.mark.parametrize("backend", ["triton"])
def test_qkv_rope_rows_and_tables(backend):
    dtype = served("qkv_rope", backend)
    op = registry.resolve("qkv_rope", "cuda:0", dtype, backend=backend)
    lin = registry.resolve("linear_act", "cuda:0", dtype, backend=backend)
    torch.manual_seed(1)
    x = rnd(2, 48, 48, 256, dtype=dtype)
    w, b = rnd(768, 256, dtype=dtype, scale=0.06), rnd(768, dtype=dtype)
    cos, sin = rope_tables(24, 24, 64, 10000.0, "cuda:0")
    with torch.inference_mode():
        q, k, v = op(x, w, b, cos, sin, 24)
        q1, k1, v1 = op(x[1:2], w, b, cos, sin, 24)
        assert torch.equal(q[1:2], q1) and torch.equal(k[1:2], k1) and torch.equal(v[1:2], v1)
        assert torch.equal(v, lin(x, w[512:], b[512:], "none").view(v.shape))
        ruler = eager.qkv_rope(x.double(), w.double(), b.double(), cos, sin, 24)
        wrong_cos, wrong_sin = rope_tables(24, 72, 64, 10000.0, "cuda:0")
        broken = op(x, w, b, wrong_cos, wrong_sin, 24)
    for got, bad, ref in zip((q, k), broken[:2], ruler[:2]):
        assert far(mean_err(bad, ref), mean_err(got, ref))


@pytest.mark.parametrize("backend", ["triton"])
@pytest.mark.parametrize("channels", [256, 1024, 1000])
def test_add_layer_norm_sum_and_rows(backend, channels):
    dtype = served("add_layer_norm", backend)
    op = registry.resolve("add_layer_norm", "cuda:0", dtype, backend=backend)
    torch.manual_seed(2)
    x, r = rnd(5, 77, channels, dtype=dtype, scale=4), rnd(5, 77, channels, dtype=dtype)
    w, b = rnd(channels, dtype=dtype), rnd(channels, dtype=dtype, scale=3)
    with torch.inference_mode():
        s, y = op(x, r, w, b, 1e-6)
        assert torch.equal(s, x + r)
        s1, y1 = op(x[2:3], r[2:3], w, b, 1e-6)
        assert torch.equal(s[2:3], s1) and torch.equal(y[2:3], y1)
        ruler = eager.add_layer_norm(
            s.double(), torch.zeros_like(s, dtype=torch.float64), w.double(), b.double(), 1e-6
        )[1]
        broken = op(x, r, b, w, 1e-6)[1]
    assert far(mean_err(broken, ruler), mean_err(y, ruler))
