"""No engine runs in the machine's process: entries that load one move to a spawned host."""

import pytest

from kohakugbc.engines import builtin, registry

assert builtin.__name__


def test_a_directly_named_engine_moves_to_the_local_host():
    if ("vlm", "vllm") not in registry.HOSTED or ("detect", "torch") not in registry.HOSTED:
        pytest.skip("vLLM or torch is not installed: their entries are absent")
    engines = {
        "vlm": {"entry": "vllm", "model": "m"},
        "detect": {"entry": "torch", "model": "d"},
        "cpu": {"entry": "python", "workers": 2},
    }
    out = registry.hosted(engines)
    host = out["hosts"]["local"]
    assert host["engines"] == {"vlm": engines["vlm"], "detect": engines["detect"]}
    assert host["order"] == [["vlm"], ["detect"]]
    assert out["vlm"] == out["detect"] == {"entry": "remote", "hosts": ["local"]}
    assert out["cpu"] == engines["cpu"]


def test_hosted_configurations_pass_unchanged():
    engines = {
        "hosts": {"card0": {"engines": {"vlm": {"entry": "tokenomos", "model": "m"}}}},
        "vlm": {"entry": "remote", "hosts": ["card0"]},
        "cpu": {"entry": "python", "workers": 2},
    }
    assert registry.hosted(engines) is engines


def test_the_local_name_is_taken_once():
    if ("vlm", "vllm") not in registry.HOSTED:
        pytest.skip("vLLM is not installed: its entry is absent")
    engines = {"hosts": {"local": {"engines": {}}}, "vlm": {"entry": "vllm", "model": "m"}}
    with pytest.raises(ValueError, match="local"):
        registry.hosted(engines)
