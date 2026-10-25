"""CUDA graphs: captures sharing a pool hold about the largest pass (one capture stream per
device), and on several cards in one process each card's ``Graphs`` captures on its own card
(a capture on the second card after one on the first succeeds, replays the right values, and
leaves the current device as it was)."""

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.kernels import graphs  # noqa: E402
from kohakugbc_models.kernels.graphs import Graphs  # noqa: E402

CUDA = torch.cuda.is_available()
TWO = CUDA and torch.cuda.device_count() >= 2


def passes(mib: int):
    """A pass with two temporaries of ``mib`` MiB."""
    n = mib << 18

    def fn(x):
        big = x.repeat(n // x.numel())
        return (big * 2 + 1).sum().reshape(1)

    return fn


def pool_growth_mib() -> float:
    """Reserved memory added by five captures of 256 / 128 / 64 / 256 / 128 MiB passes into
    one shared pool (warm-ups included)."""
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved(0)
    pool = torch.cuda.graph_pool_handle()
    x = torch.ones(1024, device="cuda:0")
    kept = [Graphs(passes(m), pool=pool) for m in (256, 128, 64, 256, 128)]
    for g in kept:
        g.warm(1, x)
    torch.cuda.synchronize(0)
    grown = (torch.cuda.memory_reserved(0) - before) / 2**20
    del kept
    torch.cuda.empty_cache()
    return grown


@pytest.mark.skipif(not CUDA, reason="needs a CUDA card")
def test_graphs_sharing_a_pool_hold_the_largest_pass(monkeypatch):
    torch.cuda.set_device(0)
    # RTX 5090: 772 MiB with one capture stream per device; 2508 MiB with a new stream per
    # capture (the allocator reuses a pool's freed blocks only on the stream that freed them)
    shared = pool_growth_mib()
    monkeypatch.setattr(graphs, "capture_stream", lambda: torch.cuda.Stream())
    fresh = pool_growth_mib()
    assert shared < 1200 < fresh, (shared, fresh)


@pytest.mark.skipif(not TWO, reason="needs two CUDA cards")
def test_a_capture_per_card():
    for card in (0, 1):
        torch.cuda.set_device(card)
        graphs = Graphs(lambda x: x * 2 + 1, clone=(0,))
        x = torch.arange(8, dtype=torch.float32, device=f"cuda:{card}")
        graphs.warm(8, x)
        assert graphs.graphs[8] is not None, f"capture on cuda:{card} failed"
        assert torch.cuda.current_device() == card
        out = graphs(8, x + 1)
        assert out.device == torch.device(f"cuda:{card}")
        assert torch.equal(out.cpu(), (torch.arange(8, dtype=torch.float32) + 1) * 2 + 1)
