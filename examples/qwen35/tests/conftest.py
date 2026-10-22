"""Shared fixtures: the checkpoint at ``QWEN35_MODEL`` loaded once, and GBC requests recorded
from KohakuGBC's test trace (the program run on its replayed replies)."""

import gc
import os
from pathlib import Path

import pytest
import torch

from kohakugbc.io.sources import TraceJobs
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference
from qwen35.model.forward import Model

TRACE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "traces" / "text"
MODEL = os.environ.get("QWEN35_MODEL", "")


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[tuple[dict, str]] = []

    def chat(self, request: dict) -> str:
        reply = super().chat(request)
        self.requests.append((request, reply))
        return reply


@pytest.fixture(scope="session")
def recorded() -> list[tuple[dict, str]]:
    """(request, recorded reply) of every chat the program asks on the trace's jobs."""
    models = Capture(Trace(TRACE))
    for item in TraceJobs(TRACE):
        reference(item.inputs, models)
    return models.requests


@pytest.fixture(scope="module")
def model():
    """Loaded for one module, then freed: the unit tests load their own."""
    if not MODEL:
        pytest.skip("QWEN35_MODEL is not set")
    loaded = Model(MODEL, "cuda:0")
    yield loaded
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
