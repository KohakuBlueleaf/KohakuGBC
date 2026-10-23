"""Decode steps (one new token per sequence) as captured CUDA graphs, one per batch bucket.

Every input lives in a static buffer; a step copies its rows in and replays the smallest bucket
that holds them. Rows past the step's sequences point at the scratch slot and the scratch page
(their results are discarded). The op choices for each bucket are bound by a warm-up forward
before its capture.
"""

import torch

from qwen35.model.forward import Step


def buckets(max_batch: int) -> list[int]:
    out, b = [], 1
    while b < max_batch:
        out.append(b)
        b = b * 2 if b < 8 else b + 8
    out.append(max_batch)
    return sorted(set(out))


class DecodeGraphs:
    def __init__(self, model, ops, pools, max_batch: int, max_pages: int):
        dev = model.device
        self.model, self.ops, self.pools = model, ops, pools
        self.sizes = buckets(max_batch)
        n = max_batch
        self.ids = torch.zeros(n, dtype=torch.long, device=dev)
        self.positions = torch.zeros(3, n, dtype=torch.long, device=dev)
        self.slots = torch.full((n,), pools.scratch_slot, dtype=torch.int32, device=dev)
        self.pages = torch.zeros(n, max_pages, dtype=torch.int32, device=dev)
        self.ctx = torch.ones(n, dtype=torch.int32, device=dev)
        self.write = torch.zeros(n, dtype=torch.long, device=dev)
        # bucket -> (graph, its logits, its step: the graph reads the step's tensors)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, Step]] = {}

    def _step(self, b: int) -> Step:
        cu = torch.arange(b + 1, dtype=torch.int32, device=self.ids.device)
        return Step(
            ids=self.ids[:b],
            positions=self.positions[:, :b],
            cu=cu,
            slots=self.slots[:b],
            pages=self.pages[:b],
            ctx=self.ctx[:b],
            write=self.write[:b],
            last=torch.arange(b, device=self.ids.device),
            max_len=1,
        )

    def capture(self) -> None:
        self.ops.graph = True
        try:
            self._capture()
        finally:
            self.ops.graph = False

    def _capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(self.ids.device)
        for b in sorted(self.sizes, reverse=True):
            step = self._step(b)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.model.forward(self.ops, self.pools, step)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self.model.forward(self.ops, self.pools, step)
            self.graphs[b] = (graph, out, step)
        torch.cuda.synchronize(self.ids.device)

    def fits(self, n: int) -> bool:
        return n <= self.sizes[-1]

    def run(self, ids, positions, slots, pages, ctx, write) -> torch.Tensor:
        """Logits of the ``n`` given rows (device tensors of the step's sequences)."""
        n = ids.shape[0]
        b = next(s for s in self.sizes if s >= n)
        self.ids[:n].copy_(ids)
        self.positions[:, :n].copy_(positions)
        self.slots[:n].copy_(slots)
        self.pages[:n].copy_(pages)
        self.ctx[:n].copy_(ctx)
        self.write[:n].copy_(write)
        if b > n:
            self.slots[n:b].fill_(self.pools.scratch_slot)
            self.pages[n:b].zero_()
            self.ctx[n:b].fill_(1)
            self.write[n:b].zero_()
        graph, out, _ = self.graphs[b]
        graph.replay()
        return out[:n]
