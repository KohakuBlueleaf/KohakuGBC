"""A callable replayed from CUDA graphs: one graph per key (one static input shape), captured
on the key's first call after an uncaptured warm-up that tunes and builds every op under the
``graph`` tuning metric; all graphs of one ``Graphs`` share one memory pool (replays never
overlap). A key whose capture fails (an op that synchronizes with the host) runs uncaptured,
logged once."""

import logging

import torch

from kohakugbc_models.kernels import tuning

log = logging.getLogger(__name__)


class Graphs:
    """``fn(*inputs)`` replayed per key. Inputs are copied into the graph's static tensors; the
    outputs at the ``clone`` indices come back cloned (they must outlive the next replay), the
    rest as the graph's own tensors (read before the next replay)."""

    def __init__(self, fn, clone: tuple[int, ...] = ()):
        self.fn, self.clone = fn, set(clone)
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle()

    def capture(self, inputs: tuple) -> tuple:
        static = tuple(t.clone() for t in inputs)
        with tuning.metric("graph"):
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

    def __call__(self, key, *inputs):
        """The outputs as ``fn`` returns them: a tuple, or one tensor (cloned when ``clone``
        holds 0)."""
        if key not in self.graphs:
            try:
                self.graphs[key] = self.capture(inputs)
            except RuntimeError as err:
                log.warning("graph capture of %s failed, running uncaptured: %s", key, err)
                self.graphs[key] = None
        if self.graphs[key] is None:
            return self.fn(*inputs)
        graph, static, outputs = self.graphs[key]
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        graph.replay()
        if torch.is_tensor(outputs):
            return outputs.clone() if 0 in self.clone else outputs
        return tuple(o.clone() if i in self.clone else o for i, o in enumerate(outputs))
