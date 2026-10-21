"""Pseudo engines: Togeari's simulated engines answering model nodes from a recorded trace,
with the trace's latencies. They test the program and the machine, never a model."""

from kohakugbc.engines.replay.entries import models_of

__all__ = ["models_of"]
