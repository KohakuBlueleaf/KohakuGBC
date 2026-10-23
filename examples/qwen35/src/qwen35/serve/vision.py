"""The vision stage: prepared requests with images pass through here before the engine. A thread
of its own runs the vision tower on a CUDA stream of its own over every request waiting (up to
``max_patches`` patches per call), so image embeddings are computed while the engine steps.
Each request leaves with its embeddings and the event that marks them done; the engine's stream
waits on that event before it reads them (no host sync).

A request whose ``images`` snapshot is already stored skips the tower (its prefill resumes after
the images).
"""

import queue
import threading
import time
import traceback

import torch


class VisionStage:
    def __init__(self, model, pools, forward, max_patches: int = 32768, log=None):
        self.model, self.pools, self.forward = model, pools, forward
        self.max_patches = max_patches
        self.log = log
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.stream = torch.cuda.Stream(model.device)
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        self.failure_text = ""
        self.counts = {"vision_calls": 0, "images": 0, "patches": 0, "vision_skipped": 0}
        self.times = {"vision_gpu": 0.0, "vision_cpu": 0.0}

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="qwen35-vision", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.inbox.put(None)
        if self.thread is not None:
            self.thread.join(timeout=30)

    def add(self, seq) -> None:
        self.inbox.put(seq)
        if self.failure is not None:
            self._drain()

    def _drain(self) -> None:
        """A dead stage answers every queued request with its failure."""
        while True:
            try:
                seq = self.inbox.get_nowait()
            except queue.Empty:
                return
            if seq is not None:
                seq.finish("unit", self.failure)

    def _batch(self) -> list:
        first = self.inbox.get()
        if first is None:
            return []
        out, patches = [first], int(first.prep.pixels.shape[0])
        while patches < self.max_patches:
            try:
                seq = self.inbox.get_nowait()
            except queue.Empty:
                break
            if seq is None:
                self.stopping.set()
                break
            out.append(seq)
            patches += int(seq.prep.pixels.shape[0])
        return out

    def _run(self) -> None:
        try:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                while not self.stopping.is_set():
                    batch = self._batch()
                    todo = []
                    for seq in batch:
                        if seq.state["cancelled"]:
                            self.forward(seq)
                        elif self._covered(seq):
                            self.counts["vision_skipped"] += 1
                            self.forward(seq)
                        else:
                            todo.append(seq)
                    if todo:
                        self._encode(todo)
        except BaseException as exc:  # noqa: BLE001
            self.failure_text = "".join(traceback.format_exception(exc))
            if self.log is not None:
                self.log.write({"kind": "failure", "stage": "vision", "error": self.failure_text})
            traceback.clear_frames(exc.__traceback__)
            self.failure = exc
            self._drain()

    def _covered(self, seq) -> bool:
        """The request's ``images`` snapshot (after its last image) is stored."""
        end = seq.prep.image_end
        return any(
            length >= end and key in self.pools.snapshots for length, key in seq.prep.boundaries
        )

    def _encode(self, todo: list) -> None:
        t0 = time.perf_counter()
        start, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        pixels = torch.cat([s.prep.pixels for s in todo]).to(self.model.device, non_blocking=True)
        grid = torch.cat([s.prep.grid for s in todo])
        embeds = self.model.vision(pixels, grid)
        sizes = [int((s.prep.grid.prod(-1) // 4).sum()) for s in todo]
        done.record(self.stream)
        for seq, part in zip(todo, embeds.split(sizes)):
            seq.embeds, seq.vision_done = part, done
        self.counts["vision_calls"] += 1
        self.counts["images"] += int(grid.shape[0])
        self.counts["patches"] += int(pixels.shape[0])
        self.times["vision_cpu"] += time.perf_counter() - t0
        for seq in todo:
            self.forward(seq)
        done.synchronize()
        ms = start.elapsed_time(done)
        self.times["vision_gpu"] += ms / 1e3
        if self.log is not None:
            self.log.write(
                {
                    "kind": "vision",
                    "images": int(grid.shape[0]),
                    "patches": int(pixels.shape[0]),
                    "gpu_ms": round(ms, 3),
                }
            )
