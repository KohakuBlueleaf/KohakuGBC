"""Every backend against the eager reference, on packed sequences of mixed lengths in shuffled
pool slots: outputs, the states written back, and slots nobody runs untouched. Each check is
shown able to fail (a wrong initial state, a wrong page table)."""

import pytest
import torch

from qwen35.ops import REGISTRY, UNAVAILABLE, Ops
from qwen35.ops import backends as _backends  # noqa: F401

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

DEV = "cuda:0"
BF = torch.bfloat16


def run(op, backend, *args):
    return REGISTRY[op][backend](*args)


def cu_of(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=DEV)


def close(a, b, tol):
    """Relative error ``|a - b| / |b|`` over the whole tensor within ``tol``."""
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item() <= tol


# ---- delta -------------------------------------------------------------------------------------


def delta_case(lengths, hk=4, hv=8, dk=128, dv=128, pool=9, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = sum(lengths)

    def r(*s, scale=1.0, dtype=BF):
        return (torch.randn(*s, generator=g, device=DEV) * scale).to(dtype)

    args = dict(
        q=r(n, hk, dk),
        k=r(n, hk, dk),
        v=r(n, hv, dv),
        a=r(n, hv),
        b=r(n, hv),
        a_log=r(hv, scale=0.5, dtype=torch.float32),
        dt_bias=r(hv, dtype=torch.float32),
    )
    state = r(pool, hv, dk, dv, scale=0.3, dtype=torch.float32)
    slots = torch.tensor([7, 0, 4, 2][: len(lengths)], dtype=torch.int32, device=DEV)
    return args, state, slots, cu_of(lengths), max(lengths)


def call_delta(backend, args, state, slots, cu, max_len):
    a = args
    return run(
        "delta",
        backend,
        a["q"],
        a["k"],
        a["v"],
        a["a"],
        a["b"],
        a["a_log"],
        a["dt_bias"],
        state,
        slots,
        cu,
        max_len,
    )


@pytest.mark.parametrize("lengths", [[1, 1, 1, 1], [3, 1, 16, 2], [70, 1, 130, 5]])
@pytest.mark.parametrize("backend", ["fla", "triton"])
def test_delta_matches_reference(lengths, backend):
    if ("delta", backend) in UNAVAILABLE:
        pytest.skip(UNAVAILABLE[("delta", backend)])
    args, state, slots, cu, max_len = delta_case(lengths)
    ref_state, got_state = state.clone(), state.clone()
    ref = call_delta("eager", args, ref_state, slots, cu, max_len)
    got = call_delta(backend, args, got_state, slots, cu, max_len)
    assert close(got, ref, 2e-2)
    assert close(got_state, ref_state, 1e-2)
    others = [i for i in range(state.shape[0]) if i not in slots.tolist()]
    assert torch.equal(got_state[others], state[others])
    # The check sees a sequence started from the wrong state
    wrong = state.clone()
    wrong[slots[-1]] = 0
    bad = call_delta(backend, args, wrong, slots, cu, max_len)
    assert not close(bad, ref, 2e-2)


# ---- conv --------------------------------------------------------------------------------------


@pytest.mark.parametrize("lengths", [[1, 1, 1], [5, 1, 2], [64, 3, 1]])
def test_conv_matches_reference(lengths):
    g = torch.Generator(device=DEV).manual_seed(1)
    n, c, w = sum(lengths), 96, 4
    x = torch.randn(n, c, generator=g, device=DEV).to(BF)
    weight = torch.randn(c, w, generator=g, device=DEV).to(BF)
    state = torch.randn(6, c, w - 1, generator=g, device=DEV).to(BF)
    slots = torch.tensor([5, 1, 3][: len(lengths)], dtype=torch.int32, device=DEV)
    cu = cu_of(lengths)
    ref_state, got_state = state.clone(), state.clone()
    ref = run("conv", "eager", x, weight, ref_state, slots, cu)
    got = run("conv", "torch", x, weight, got_state, slots, cu)
    assert close(got, ref, 1e-2)
    assert torch.equal(got_state, ref_state)
    wrong = state.clone()
    wrong[slots[0]] = 0
    bad = run("conv", "torch", x, weight, wrong, slots, cu)
    assert not close(bad, ref, 1e-2)


# ---- attention ---------------------------------------------------------------------------------


def attention_case(cached, new, heads=8, kvh=2, hd=256, page=16, seed=2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = sum(new)
    total = [c + k for c, k in zip(cached, new)]
    max_pages = max(-(-t // page) for t in total) + 1
    n_pages = 1 + sum(-(-t // page) for t in total)
    order = torch.randperm(n_pages - 1, generator=torch.Generator().manual_seed(seed)) + 1
    tables, used = torch.zeros(len(new), max_pages, dtype=torch.int32), 0
    for i, t in enumerate(total):
        k = -(-t // page)
        tables[i, :k] = order[used : used + k]
        used += k
    cache_k = torch.randn(n_pages, page, kvh, hd, generator=g, device=DEV).to(BF)
    cache_v = torch.randn(n_pages, page, kvh, hd, generator=g, device=DEV).to(BF)
    rows = []
    for i, (c, t) in enumerate(zip(cached, total)):
        pos = torch.arange(c, t)
        rows.append(tables[i, pos // page].long() * page + pos % page)
    q = torch.randn(n, heads, hd, generator=g, device=DEV).to(BF)
    k = torch.randn(n, kvh, hd, generator=g, device=DEV).to(BF)
    v = torch.randn(n, kvh, hd, generator=g, device=DEV).to(BF)
    return (
        q,
        k,
        v,
        cache_k,
        cache_v,
        tables.to(DEV),
        torch.tensor(total, dtype=torch.int32, device=DEV),
        cu_of(new),
        torch.cat(rows).to(DEV),
        max(new),
        hd**-0.5,
    )


@pytest.mark.parametrize(
    ("cached", "new"),
    [([5, 0, 40], [1, 1, 1]), ([17, 3, 0, 100], [1, 7, 33, 2]), ([0, 0], [60, 129])],
)
@pytest.mark.parametrize("backend", ["sdpa", "triton"])
def test_attention_matches_reference(cached, new, backend):
    if ("attention", backend) in UNAVAILABLE:
        pytest.skip(UNAVAILABLE[("attention", backend)])
    args = attention_case(cached, new)
    ref_args = list(args)
    ref_args[3], ref_args[4] = args[3].clone(), args[4].clone()
    ref = run("attention", "eager", *ref_args)
    got_args = list(args)
    got_args[3], got_args[4] = args[3].clone(), args[4].clone()
    got = run("attention", backend, *got_args)
    assert close(got, ref, 2e-2)
    assert torch.equal(got_args[3], ref_args[3])
    # The check sees a wrong page table
    bad_args = list(got_args)
    bad_args[5] = args[5].roll(1, dims=1)
    bad = run("attention", backend, *bad_args)
    assert not close(bad, ref, 2e-2)


def test_ops_bind_once_per_key_by_race():
    ops = Ops({"delta": ["triton", "fla"]}, race=True)
    args, state, slots, cu, max_len = delta_case([1, 1, 1, 1])
    a = args
    call = (
        a["q"],
        a["k"],
        a["v"],
        a["a"],
        a["b"],
        a["a_log"],
        a["dt_bias"],
        state,
        slots,
        cu,
        max_len,
    )
    before = state.clone()
    ops("delta", *call)
    (key,) = list(ops.choices)
    assert key == ("delta", "n4-t1")
    assert set(ops.choices[key]["ms"]) <= {"triton", "fla"}
    # The race ran on scratch copies: the pool advanced exactly once
    once = before.clone()
    call_delta("eager", args, once, slots, cu, max_len)
    assert close(state, once, 1e-2)
