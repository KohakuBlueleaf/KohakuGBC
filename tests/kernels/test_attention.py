"""The attention kernels' exact properties, on every GPU backend that serves them here:

- batched equals single, bit for bit (programs never share work across the batch);
- the windowed ViT op on the grid equals the same op on each window cut out as its own grid;
- a broken variant (swapped box-bias tables, RoPE tables of the wrong window grid) lands far
  from fp64 next to the correct call (relative to it: the precision itself is reviewed in the
  bench's profiles, not judged here).
"""

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.detect.sam3.vision import rope_tables  # noqa: E402
from kohakugbc_models.kernels import registry, tuning  # noqa: E402
from kohakugbc_models.kernels.backends.eager import attention as eager  # noqa: E402

ENTRIES = []
try:
    from kohakugbc_models.kernels.backends.triton.attention import entry as triton_entry

    ENTRIES.append(triton_entry)
except ImportError:
    pass
try:
    from kohakugbc_models.kernels.backends.tilelang.attention import entry as tilelang_entry

    ENTRIES.append(tilelang_entry)
except ImportError:
    pass

pytestmark = pytest.mark.gpu
CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def one_tile(tmp_path, monkeypatch):
    """Every problem binds its first tile, untimed: tuned per batch bucket, a batched and a
    single call could run different tiles, which sum the keys in a different order."""
    monkeypatch.setenv("KOHAKUGBC_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuning, "_CHOSEN", {})
    monkeypatch.setattr(tuning, "_TABLES", {})
    registry.load_backends()
    for module in ENTRIES:
        monkeypatch.setattr(module, "_BOUND", {})
    tuning.set_mode("off")
    yield
    tuning.set_mode("table")


def backends(op: str, dtype: torch.dtype) -> list[str]:
    if not CUDA:
        return []
    reasons = registry.available(op, "cuda:0", dtype)
    return [b for b in ("triton", "tilelang") if reasons[b] is None]


def dtype_here() -> torch.dtype:
    if CUDA and torch.cuda.get_device_capability(0)[0] < 8:
        return torch.float16
    return torch.bfloat16


def pinned(op: str, backend: str, dtype: torch.dtype):
    return registry.resolve(op, "cuda:0", dtype, backend=backend)


def rnd(*shape, dtype, scale=1.0):
    return (torch.randn(*shape, device="cuda:0") * scale).to(dtype)


def far(err_broken: float, err_ok: float) -> bool:
    return err_broken > 100 * max(err_ok, 1e-30)


def mean_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().mean().item()


@pytest.mark.parametrize("backend", ["triton", "tilelang"])
def test_vit_windows_and_batches(backend):
    dtype = dtype_here()
    if backend not in backends("vit_attention", dtype):
        pytest.skip(f"{backend} does not serve vit_attention here")
    op = pinned("vit_attention", backend, dtype)
    torch.manual_seed(0)
    q, k, v = (rnd(3, 48, 48, 4, 64, dtype=dtype) for _ in range(3))
    cos, sin = rope_tables(24, 24, 64, 10000.0, "cuda:0")
    with torch.inference_mode():
        full = op(q, k, v, cos, sin, 24)
        single = torch.cat(
            [op(q[i : i + 1], k[i : i + 1], v[i : i + 1], cos, sin, 24) for i in range(3)]
        )
        assert torch.equal(full, single)
        for wy in range(2):
            for wx in range(2):
                cut = (slice(None), slice(24 * wy, 24 * wy + 24), slice(24 * wx, 24 * wx + 24))
                alone = op(
                    q[cut].contiguous(), k[cut].contiguous(), v[cut].contiguous(), cos, sin, 24
                )
                assert torch.equal(full[cut], alone), (wy, wx)
        ruler = eager.vit_attention(q.double(), k.double(), v.double(), cos, sin, 24)
        wrong_cos, wrong_sin = rope_tables(24, 72, 64, 10000.0, "cuda:0")
        broken = op(q, k, v, wrong_cos, wrong_sin, 24)
    assert far(mean_err(broken, ruler), mean_err(full, ruler))


@pytest.mark.parametrize("backend", ["triton", "tilelang"])
def test_rpb_batches_and_table_axes(backend):
    dtype = dtype_here()
    if backend not in backends("attention_rpb", dtype):
        pytest.skip(f"{backend} does not serve attention_rpb here")
    op = pinned("attention_rpb", backend, dtype)
    torch.manual_seed(1)
    q = rnd(3, 201, 8, 32, dtype=dtype).transpose(1, 2)
    k = rnd(3, 48 * 48, 8, 32, dtype=dtype).transpose(1, 2)
    v = rnd(3, 48 * 48, 8, 32, dtype=dtype).transpose(1, 2)
    # The tables in the layout the decoder makes them: a padded permute of [B, Q, grid, heads]
    y = rnd(3, 201, 48, 8, dtype=dtype, scale=3).permute(0, 3, 1, 2)
    x = rnd(3, 201, 48, 8, dtype=dtype, scale=3).permute(0, 3, 1, 2)
    assert not y.is_contiguous()
    with torch.inference_mode():
        full = op(q, k, v, y, x)
        single = torch.cat(
            [
                op(q[i : i + 1], k[i : i + 1], v[i : i + 1], y[i : i + 1], x[i : i + 1])
                for i in range(3)
            ]
        )
        assert torch.equal(full, single)
        ruler = eager.attention_rpb(q.double(), k.double(), v.double(), y.double(), x.double())
        broken = op(q, k, v, x, y)
        # The same values contiguous give the same output: the tables' strides are honoured
        assert torch.equal(full, op(q, k, v, y.contiguous(), x.contiguous()))
    assert far(mean_err(broken, ruler), mean_err(full, ruler))


@pytest.mark.parametrize("backend", ["triton", "tilelang"])
def test_masked_keys_are_excluded(backend):
    dtype = dtype_here()
    if backend not in backends("attention", dtype):
        pytest.skip(f"{backend} does not serve attention here")
    op = pinned("attention", backend, dtype)
    torch.manual_seed(2)
    q = rnd(2, 300, 8, 32, dtype=dtype).transpose(1, 2)
    k = rnd(2, 32, 8, 32, dtype=dtype).transpose(1, 2)
    v = rnd(2, 32, 8, 32, dtype=dtype).transpose(1, 2)
    mask = torch.zeros(2, 1, 1, 32, dtype=torch.bool, device="cuda:0")
    mask[0, ..., :5] = True
    mask[1, ..., :17] = True
    with torch.inference_mode():
        out = op(q, k, v, mask)
        # Keys past the valid ones never reach the output: changing them changes nothing
        k2, v2 = k.clone(), v.clone()
        k2[0, :, 5:] = 100
        v2[0, :, 5:] = -100
        assert torch.equal(out[0], op(q, k2, v2, mask)[0])
