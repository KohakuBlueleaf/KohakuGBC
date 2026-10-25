"""The ``detect`` slot's native SAM 3 as the entry builds it (no weights loaded): the checkpoint
it loads, the crop cap the batch size sets, ``fp8: auto`` on a device without fp8."""

from pathlib import Path

import pytest

from kohakugbc.engines import registry
from kohakugbc.machine import build_parts  # noqa: F401 (registers the entries)

TRACE = Path(__file__).resolve().parents[1] / "fixtures" / "traces" / "text"


def detector(**settings):
    slots = registry.build(
        {
            "vlm": {"entry": "replay", "trace": str(TRACE)},
            "detect": {"entry": "torch", "model": "facebook/sam3", "impl": "native", **settings},
            "embed": {"entry": "replay", "trace": str(TRACE)},
            "cpu": {"entry": "python"},
        }
    )
    engine = slots["detect"].units[0].engine
    return next(iter(engine.models.values())).model


def test_the_entry_loads_the_named_checkpoint_with_its_caps():
    model = detector(
        devices=["cpu"], max_batch=8, max_crops=16, fp8="auto", vision_graph=True, budget_gb=None
    )
    det = model.detector
    assert model.model_name == det.model_name == "facebook/sam3"
    # A batch of 8 requests is 8 crops: no vision pass runs more
    assert det.max_crops == 8 and model.max_batch == 8
    assert det.fp8 is None and det.vision_graph


def test_a_budget_without_a_measured_plan_is_refused():
    with pytest.raises(ValueError, match="memory plan"):
        detector(devices=["cpu"], budget_gb=2.0)
