"""The kernel registry: priority order, supports / probe / failure reasons, the fallback that
rebinds every live op of the same query on a failed backend, and what never falls back
(an out-of-memory, a sticky device error, a pinned backend)."""

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.kernels import registry  # noqa: E402
from kohakugbc_models.kernels.errors import BackendUnavailable  # noqa: E402


@pytest.fixture
def fresh(monkeypatch):
    """An empty registry with a two-backend priority on the CPU."""
    monkeypatch.setattr(registry, "IMPLS", {})
    monkeypatch.setattr(registry, "FAILED", {})
    monkeypatch.setattr(registry, "UNAVAILABLE", {})
    monkeypatch.setattr(registry, "PROBES", {})
    monkeypatch.setattr(registry, "_PROBED", {})
    monkeypatch.setattr(registry, "_LOADED", [True])
    monkeypatch.setitem(registry.PRIORITY, "cpu", ("triton", "eager"))


def test_first_supporting_backend_binds(fresh):
    registry.register("op", "triton", lambda x: x + 1, supports=lambda q: None)
    registry.register("op", "eager", lambda x: x + 2)
    bound = registry.resolve("op", "cpu", torch.float32)
    assert bound.backend == "triton" and bound(1) == 2


def test_supports_and_probe_reasons_skip_a_backend(fresh):
    registry.register("op", "triton", lambda x: x, supports=lambda q: "no fp32 here")
    registry.register("op", "eager", lambda x: x)
    assert registry.resolve("op", "cpu", torch.float32).backend == "eager"
    registry.register("op", "triton", lambda x: x)
    registry.register_probe("triton", lambda device: "no driver")
    reasons = registry.available("op", "cpu", torch.float32)
    assert "no driver" in reasons["triton"] and reasons["eager"] is None


def test_failure_rebinds_every_live_op_of_the_query(fresh):
    calls = []

    def broken(x):
        calls.append("triton")
        raise RuntimeError("ptxas failed")

    registry.register("op", "triton", broken)
    registry.register("op", "eager", lambda x: x * 10)
    a = registry.resolve("op", "cpu", torch.float32)
    b = registry.resolve("op", "cpu", torch.float32)
    other = registry.resolve("op", "cpu", torch.float16)
    assert a(2) == 20
    assert a.backend == b.backend == "eager" and other.backend == "triton"
    assert calls == ["triton"]
    # A later resolve skips the backend that failed this query here
    assert registry.resolve("op", "cpu", torch.float32).backend == "eager"


def test_out_of_memory_and_sticky_errors_propagate(fresh):
    def oom(x):
        raise torch.OutOfMemoryError("CUDA out of memory")

    def sticky(x):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    registry.register("oom", "triton", oom)
    registry.register("oom", "eager", lambda x: x)
    registry.register("sticky", "triton", sticky)
    registry.register("sticky", "eager", lambda x: x)
    with pytest.raises(torch.OutOfMemoryError):
        registry.resolve("oom", "cpu", torch.float32)(1)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        registry.resolve("sticky", "cpu", torch.float32)(1)
    assert registry.FAILED == {}


def test_a_pinned_backend_does_not_fall_back(fresh):
    def broken(x):
        raise RuntimeError("build failed")

    registry.register("op", "triton", broken)
    registry.register("op", "eager", lambda x: x)
    with pytest.raises(BackendUnavailable):
        registry.resolve("op", "cpu", torch.float32, backend="triton")(1)
