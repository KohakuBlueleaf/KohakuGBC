"""The tuner: a timed shortlist stores its winner with every timing and failure; a stored
winner binds without timing; a config over the working-memory bound or failing to build is
skipped; every config failing raises ``ConfigsFailed``; ``off`` never times."""

import json
import time

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.kernels import tuning  # noqa: E402
from kohakugbc_models.kernels.errors import ConfigsFailed  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUGBC_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuning, "_CHOSEN", {})
    monkeypatch.setattr(tuning, "_TABLES", {})
    tuning.set_mode("table")
    return tmp_path


def sleeper(ms: float):
    return lambda: time.sleep(ms / 1e3)


def test_the_fastest_is_stored_and_reused(cache):
    configs = [{"tile": 64}, {"tile": 128}, {"tile": 32}]
    launches = []

    def launch(config):
        launches.append(config["tile"])
        return sleeper({64: 6, 128: 1, 32: 3}[config["tile"]])

    won = tuning.select("k", "cpu", "key-a", lambda: configs, launch, repeat=2)
    assert won == {"tile": 128}
    table = json.loads(next(cache.rglob("k.json")).read_text(encoding="utf-8"))
    assert table["key-a"]["config"] == {"tile": 128}
    assert set(table["key-a"]["timings_ms"]) == {"tile=64", "tile=128", "tile=32"}
    # A new process (fresh in-memory state) binds the stored winner without timing
    tuning._CHOSEN.clear()
    tuning._TABLES.clear()
    launches.clear()
    assert tuning.select("k", "cpu", "key-a", lambda: configs, launch) == {"tile": 128}
    assert launches == []


def test_failing_configs_are_skipped_and_all_failing_raises(cache):
    def launch(config):
        if config["tile"] != 16:
            raise RuntimeError("out of shared memory")
        return sleeper(1)

    configs = [{"tile": 128}, {"tile": 16}]
    assert tuning.select("k", "cpu", "key-b", lambda: configs, launch, repeat=1) == {"tile": 16}
    with pytest.raises(ConfigsFailed):
        tuning.select("k", "cpu", "key-d", lambda: [{"tile": 128}, {"tile": 64}], launch)


def test_over_the_working_bound_is_skipped(cache, monkeypatch):
    used = {"lean": 100 << 20, "fat": 4 << 30}
    monkeypatch.setattr(tuning, "working_bytes", lambda fn, device: used[fn.name])

    def launch(config):
        fn = sleeper(1 if config["path"] == "fat" else 3)
        fn.name = config["path"]
        return fn

    configs = [{"path": "lean"}, {"path": "fat"}]
    assert tuning.select("k", "cpu", "key-e", lambda: configs, launch, repeat=1) == {"path": "lean"}
    table = json.loads(next(cache.rglob("k.json")).read_text(encoding="utf-8"))
    assert "working-memory bound" in table["key-e"]["failures"]["path=fat"]


def test_each_metric_keeps_its_own_winner(cache):
    """Off CUDA ``graph`` times as ``loop``, but its choice is stored and bound apart."""
    configs = [{"tile": 1}, {"tile": 2}]
    slow = {"loop": 2, "graph": 1}
    for name in ("loop", "graph"):
        with tuning.metric(name):
            won = tuning.select(
                "k",
                "cpu",
                "key-g",
                lambda: configs,
                lambda c, n=name: sleeper(3 if c["tile"] == slow[n] else 1),
                repeat=2,
            )
        assert won == {"tile": 3 - slow[name]}
    table = json.loads(next(cache.rglob("k.json")).read_text(encoding="utf-8"))
    assert set(table) == {"key-g", "key-g@graph"}


@pytest.mark.gpu
def test_graph_metric_leaves_host_time_out(cache):
    """A host-heavy call (a sleep, then one tiny kernel) loses the loop metric to a
    device-heavy one and wins the graph metric, where only device time counts."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    small = torch.ones(16, device="cuda:0")
    big = torch.randn(2048, 2048, device="cuda:0")

    def launch(config):
        if config["path"] == "host":
            return lambda: (time.sleep(0.005), small.add_(1))
        return lambda: big @ big

    configs = [{"path": "host"}, {"path": "device"}]
    with tuning.metric("loop"):
        assert tuning.select("m", "cuda:0", "k", lambda: configs, launch, repeat=5) == configs[1]
    with tuning.metric("graph"):
        assert tuning.select("m", "cuda:0", "k", lambda: configs, launch, repeat=5) == configs[0]


def test_off_mode_never_times(cache):
    tuning.set_mode("off")
    try:
        launches = []
        configs = [{"tile": 1}, {"tile": 2}]
        got = tuning.select(
            "k", "cpu", "key-f", lambda: configs, lambda c: launches.append(c) or sleeper(0)
        )
        assert got == {"tile": 1} and launches == []
    finally:
        tuning.set_mode("table")
