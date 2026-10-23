"""A callable replayed from CUDA graphs: one graph per key (one static input shape), captured
after an uncaptured warm-up that tunes and builds every op under the ``graph`` tuning metric;
all graphs of one ``Graphs`` share one memory pool (replays never overlap).

Captures run one at a time in the process and never beside the kernels' host synchronizations
(``hardware.QUIET``). They happen on a key's first call, or
ahead of time (``warm``) and then ``freeze``: a frozen ``Graphs`` captures nothing more and runs
an unseen key uncaptured, so no capture overlaps the work of other threads once a machine runs
(a host synchronization elsewhere during a capture invalidates it). A key whose capture fails
(an op that synchronizes with the host) runs uncaptured, logged once."""

import logging

import torch

from kohakugbc_models.kernels import tuning
from kohakugbc_models.kernels.hardware import QUIET

log = logging.getLogger(__name__)


class Graphs:
    """``fn(*inputs)`` replayed per key. Inputs are copied into the graph's static tensors; the
    outputs at the ``clone`` indices come back cloned (they must outlive the next replay), the
    rest as the graph's own tensors (read before the next replay)."""

    def __init__(self, fn, clone: tuple[int, ...] = ()):
        self.fn, self.clone = fn, set(clone)
        self.graphs: dict = {}
        self.frozen = False
        self.pool = torch.cuda.graph_pool_handle()

    def capture(self, inputs: tuple) -> tuple:
        static = tuple(t.clone() for t in inputs)
        with QUIET, tuning.metric("graph"):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self.fn(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # thread_local: other models' threads keep launching on their own streams
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                outputs = self.fn(*static)
        return graph, static, outputs

    def warm(self, key, *inputs) -> None:
        """Capture ``key`` now, from inputs of its shape."""
        if key not in self.graphs:
            try:
                self.graphs[key] = self.capture(inputs)
            except RuntimeError as err:
                log.warning("graph capture of %s failed, running uncaptured: %s", key, err)
                self.graphs[key] = None

    def freeze(self) -> None:
        """No more captures: unseen keys run uncaptured."""
        self.frozen = True

    def __call__(self, key, *inputs):
        """The outputs as ``fn`` returns them: a tuple, or one tensor (cloned when ``clone``
        holds 0)."""
        if key not in self.graphs:
            if self.frozen:
                return self.fn(*inputs)
            self.warm(key, *inputs)
        if self.graphs[key] is None:
            return self.fn(*inputs)
        graph, static, outputs = self.graphs[key]
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        graph.replay()
        if torch.is_tensor(outputs):
            return outputs.clone() if 0 in self.clone else outputs
        return tuple(o.clone() if i in self.clone else o for i, o in enumerate(outputs))
