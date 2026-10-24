"""The detect and embed lanes' batch formers: request counts, or each request's costs."""

from types import SimpleNamespace

import pytest
from togeari import WorkItem
from togeari.policies import BudgetFormer, SizeOrWait

pytest.importorskip("torch")

from kohakugbc.engines.torch.entries import former_of

MODEL = SimpleNamespace(max_batch=8)


def detect_item(texts: int) -> WorkItem:
    request = {"key": "k", "texts": [f"t{i}" for i in range(texts)], "box": None}
    return WorkItem("detector", {"patch": None, "request": request}, adapter=None)


def test_size_is_the_default():
    former = former_of("detector", MODEL, {}, {"crops": 4, "pairs": 16})
    assert type(former) is SizeOrWait


def test_budget_batches_one_pass_of_crops_and_pairs():
    former = former_of("detector", MODEL, {"former": "budget"}, {"crops": 4, "pairs": 16})
    assert type(former) is BudgetFormer
    items = [detect_item(n) for n in (5, 6, 4, 3, 1, 1)]
    for item in items:
        assert former.prepare(item).costs == (
            ("crops", 1.0),
            ("pairs", float(len(item.inputs["request"]["texts"]))),
        )
    # 5 + 6 + 4 pairs fit 16; the fourth request (3 pairs) would pass it: the batch stops there
    assert former.select(items, 8) == items[:3]
    # Four one-pair requests: the crop budget ends the batch at four
    ones = [detect_item(1) for _ in range(6)]
    for item in ones:
        former.prepare(item)
    assert former.select(ones, 8) == ones[:4]


def test_a_request_over_the_budget_runs_alone():
    former = former_of("detector", MODEL, {"former": "budget"}, {"crops": 4, "pairs": 16})
    big, small = detect_item(20), detect_item(2)
    former.prepare(big)
    former.prepare(small)
    assert former.select([big, small], 8) == [big]


def test_budgets_from_settings_and_the_embedder_costs():
    former = former_of(
        "embedder",
        MODEL,
        {"former": "budget", "budgets": {"chars": 10}},
        {"texts": 64, "chars": 99},
    )
    items = [WorkItem("embedder", {"text": t}, adapter=None) for t in ("abcd", "efgh", "ijk")]
    for item in items:
        former.prepare(item)
    assert former.select(items, 8) == items[:2]


def test_unknown_former_is_refused():
    with pytest.raises(ValueError):
        former_of("detector", MODEL, {"former": "bucket"}, {"crops": 4, "pairs": 16})
