"""The engine: one thread that owns the language model, the pools and every running sequence.

Each turn of its loop takes in requests (prepared, their images already encoded by the vision
stage) and cancellations, admits waiting sequences while a slot and the pages for prompt +
``max_tokens`` are free (resuming from the longest stored prefix snapshot), and runs one step:
every generating sequence's pending tokens (its sampled token and any forced ones) and, when
prefill is due, prompt chunks of admitted sequences up to ``prefill_tokens``, all in one forward.
A chunk ends at a snapshot boundary not yet stored, and the snapshot is taken after the step.
Prefill is due when the ready prompt tokens reach ``prefill_min``, when nothing generates, or
when a prompt has waited ``prefill_wait_ms``; a step of single decode tokens replays a captured
graph. While the device runs the step, the host fills the grammar masks the step's draws need.
Nothing is preempted: what a sequence may use is held from its admission.
"""

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from qwen35.model.forward import Step
from qwen35.serve.inputs import Prepared


@dataclass
class Seq:
    ticket: int
    prep: Prepared
    finish: object  # callback(result_kind, text_or_exc)
    state: dict = field(default_factory=lambda: {"cancelled": False})
    slot: int = -1
    table: list[int] = field(default_factory=list)
    cached: int = 0
    points: list[tuple[int, bytes]] = field(default_factory=list)
    embeds: torch.Tensor | None = None  # image embeddings of the prompt, until it is in
    vision_done: object = None  # the event after which ``embeds`` are ready
    pending: list[int] = field(default_factory=list)
    out: list[int] = field(default_factory=list)
    matcher: object = None
    mask_ready: bool = False
    next_position: int = 0
    t_arrive: float = 0.0


@dataclass
class Settings:
    max_seqs: int = 64
    prefill_tokens: int = 4096
    prefill_min: int = 512
    prefill_wait_ms: float = 50.0
    jump: str = "exact"  # "exact", "string" or "off" (Engine._jump)
    jump_max: int = 16
    seed: int = 0


class Engine:
    def __init__(
        self,
        model,
        ops,
        pools,
        grammars,
        tokenizer,
        settings: Settings,
        max_pages: int,
        stop: set[int],
        log=None,
    ):
        self.model, self.ops, self.pools, self.grammars = model, ops, pools, grammars
        self.tokenizer, self.s = tokenizer, settings
        self.max_pages, self.stops = max_pages, set(stop)
        self.log = log
        self.dev = model.device
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.wake = threading.Event()
        self.waiting: deque[Seq] = deque()
        self.prefilling: list[Seq] = []
        self.decoding: list[Seq] = []
        self.masks = grammars.masks(pools.state.shape[1])
        self.generator = torch.Generator(device=self.dev).manual_seed(settings.seed)
        self.graphs = None
        self.timing: deque = deque()
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.failure: BaseException | None = None
        self.failure_text = ""
        self.counts = {
            "steps": 0,
            "graph_steps": 0,
            "mixed_steps": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "generated_tokens": 0,
            "forced_tokens": 0,
            "finished": 0,
            "cancelled": 0,
            "errors": 0,
        }
        self.times = {
            "build": 0.0,
            "launch": 0.0,
            "gpu": 0.0,
            "grammar": 0.0,
            "sample": 0.0,
            "accept": 0.0,
            "vision_inline": 0.0,
        }

    # ---- outside the engine thread -----------------------------------------------------------

    def start(self, graphs=None) -> None:
        self.graphs = graphs
        self.thread = threading.Thread(target=self._run, name="qwen35-engine", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=30)

    def add(self, seq: Seq) -> None:
        self.inbox.put(("add", seq))
        self.wake.set()
        if self.failure is not None:
            self._drain()

    def _drain(self) -> None:
        """A dead loop answers every queued sequence with its failure."""
        while True:
            try:
                kind, value = self.inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "add":
                value.finish("unit", self.failure)

    def cancel(self, ticket: int) -> None:
        self.inbox.put(("cancel", ticket))
        self.wake.set()

    def load(self) -> tuple[int, int]:
        return len(self.prefilling) + len(self.decoding), len(self.waiting)

    # ---- the loop ----------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            with torch.inference_mode():
                while not self.stopping.is_set():
                    self._intake()
                    self._admit()
                    if not self._step():
                        self.wake.wait(0.01)
                        self.wake.clear()
                self._timed(wait=True)
        except BaseException as exc:  # noqa: BLE001
            self.failure_text = "".join(traceback.format_exception(exc))
            if self.log is not None:
                self.log.write({"kind": "failure", "stage": "engine", "error": self.failure_text})
            traceback.clear_frames(exc.__traceback__)
            self.failure = exc
            for seq in [*self.waiting, *self.prefilling, *self.decoding]:
                seq.finish("unit", exc)
            self.waiting.clear()
            self.prefilling.clear()
            self.decoding.clear()
            self._drain()

    def _intake(self) -> None:
        cancels = set()
        while True:
            try:
                kind, value = self.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "cancel":
                cancels.add(value)
                continue
            if value.state["cancelled"]:
                self.counts["cancelled"] += 1
                value.finish("cancelled", None)
                continue
            value.matcher = self.grammars.matcher(value.prep.schema) if value.prep.schema else None
            value.next_position = value.prep.next_position
            value.t_arrive = time.perf_counter()
            if value.embeds is not None:
                # Read on this stream after the vision stream's event; freed only after this use
                torch.cuda.current_stream().wait_event(value.vision_done)
                value.embeds.record_stream(torch.cuda.current_stream())
            self.waiting.append(value)
        if not cancels:
            return
        for group in (self.waiting, self.prefilling, self.decoding):
            for seq in list(group):
                if seq.ticket in cancels:
                    group.remove(seq)
                    self._release(seq)
                    self.counts["cancelled"] += 1
                    seq.finish("cancelled", None)

    def _release(self, seq: Seq) -> None:
        if seq.slot >= 0:
            self.pools.give_slot(seq.slot)
            self.pools.give_pages(seq.table)
            seq.slot, seq.table = -1, []
        seq.embeds = None

    # ---- admission ---------------------------------------------------------------------------

    def _admit(self) -> None:
        pools, page = self.pools, self.pools.page
        while self.waiting and len(self.prefilling) + len(self.decoding) < self.s.max_seqs:
            seq = self.waiting[0]
            prep = seq.prep
            need = len(prep.ids) + prep.max_tokens + 1
            if -(-need // page) > self.max_pages:
                self.waiting.popleft()
                seq.finish("item", ValueError(f"{need} tokens exceed the model length"))
                continue
            snap = None
            for _, key in reversed(prep.boundaries):
                snap = pools.find(key)
                if snap is not None:
                    break
            n_pages = -(-need // page) - (len(snap.pages) if snap else 0)
            if pools.pages_free() < n_pages:
                break
            slot = pools.take_slot()
            if slot is None:
                break
            pages = pools.take_pages(n_pages)
            if pages is None:
                pools.give_slot(slot)
                break
            self.waiting.popleft()
            seq.slot = slot
            if snap is not None:
                seq.table = pools.resume(snap, slot, pages)
                seq.cached = snap.length
                self.counts["cached_tokens"] += snap.length
            else:
                pools.clear_slot(slot)
                seq.table = pages
            if prep.pixels is not None and seq.embeds is None and seq.cached < prep.image_end:
                # The snapshot the vision stage counted on went: encode here
                t0 = time.perf_counter()
                seq.embeds = self.model.vision(prep.pixels, prep.grid)
                self.times["vision_inline"] += time.perf_counter() - t0
            seq.points = [
                (length, key)
                for length, key in prep.boundaries
                if length > seq.cached and pools.find(key) is None
            ]
            self.prefilling.append(seq)

    # ---- a step ------------------------------------------------------------------------------

    def _rows(self, seq: Seq, start: int, end: int) -> np.ndarray:
        t = np.arange(start, end)
        table = np.asarray(seq.table, dtype=np.int64)
        return table[t // self.pools.page] * self.pools.page + t % self.pools.page

    def _table(self, seqs: list[Seq]) -> torch.Tensor:
        out = np.zeros((len(seqs), self.max_pages), dtype=np.int32)
        for i, seq in enumerate(seqs):
            out[i, : len(seq.table)] = seq.table
        return torch.from_numpy(out).to(self.dev, non_blocking=True)

    def _chunks(self, decoding: bool) -> list[tuple[Seq, int, int]]:
        """The prompt chunks of this step, or none when prefill is not due."""
        if not self.prefilling:
            return []
        if decoding:
            ready = sum(len(s.prep.ids) - s.cached for s in self.prefilling)
            oldest = (time.perf_counter() - self.prefilling[0].t_arrive) * 1e3
            if ready < self.s.prefill_min and oldest < self.s.prefill_wait_ms:
                return []
        budget, out = self.s.prefill_tokens, []
        for seq in self.prefilling:
            if budget <= 0:
                break
            end = min(len(seq.prep.ids), seq.cached + budget)
            if seq.points:
                end = min(end, seq.points[0][0])
            out.append((seq, seq.cached, end))
            budget -= end - seq.cached
        return out

    def _step(self) -> bool:
        t0 = time.perf_counter()
        decode = [s for s in self.decoding if s.pending]
        chunks = self._chunks(bool(decode))
        if not decode and not chunks:
            return False
        ids, pos, write, cu, last, ctx = [], [], [], [0], [], []
        rows, embeds = [], []
        for seq in decode:
            k = len(seq.pending)
            ids.append(torch.tensor(seq.pending, dtype=torch.long))
            p = torch.arange(seq.next_position, seq.next_position + k)
            pos.append(p[None].expand(3, -1))
            seq.next_position += k
            write.append(self._rows(seq, seq.cached, seq.cached + k))
            seq.cached += k
            ctx.append(seq.cached)
            cu.append(cu[-1] + k)
            last.append(cu[-1] - 1)
            seq.pending = []
        image_token = self.model.shape.image_token
        for seq, s, e in chunks:
            prep = seq.prep
            ids.append(prep.ids[s:e])
            pos.append(prep.positions[:, s:e])
            write.append(self._rows(seq, s, e))
            if seq.embeds is not None:
                image = (prep.ids == image_token).nonzero().flatten()
                inside = (image >= s) & (image < e)
                if bool(inside.any()):
                    rows.append(image[inside] - s + cu[-1])
                    embeds.append(seq.embeds[inside.nonzero().flatten().to(self.dev)])
            seq.cached = e
            ctx.append(e)
            cu.append(cu[-1] + e - s)
            last.append(cu[-1] - 1)
        seqs = decode + [c[0] for c in chunks]
        n_prompt = sum(e - s for _, s, e in chunks)
        graph = not chunks and cu[-1] == len(decode) and self.graphs is not None
        graph = graph and self.graphs.fits(len(decode))
        ids_t = torch.cat(ids).to(self.dev, non_blocking=True)
        pos_t = torch.cat(pos, dim=1).to(self.dev, non_blocking=True)
        slots = torch.tensor([s.slot for s in seqs], dtype=torch.int32).to(self.dev)
        pages = self._table(seqs)
        ctx_t = torch.tensor(ctx, dtype=torch.int32).to(self.dev, non_blocking=True)
        write_t = torch.from_numpy(np.concatenate(write)).to(self.dev, non_blocking=True)
        t_build = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if graph:
            logits = self.graphs.run(ids_t, pos_t, slots, pages, ctx_t, write_t)
        else:
            step = Step(
                ids=ids_t,
                positions=pos_t,
                cu=torch.tensor(cu, dtype=torch.int32).to(self.dev),
                slots=slots,
                pages=pages,
                ctx=ctx_t,
                write=write_t,
                last=torch.tensor(last, dtype=torch.long).to(self.dev),
                max_len=max(b - a for a, b in zip(cu[:-1], cu[1:])),
                image_rows=torch.cat(rows).to(self.dev) if rows else None,
                image_embeds=torch.cat(embeds) if embeds else None,
            )
            logits = self.model.forward(self.ops, self.pools, step)
        end.record()
        # Snapshots after the step, in stream order; prompts done leave prefill
        drawn = list(decode)
        drawn_rows = list(range(len(decode)))
        for i, (seq, _, e) in enumerate(chunks):
            if seq.points and seq.points[0][0] == e:
                length, key = seq.points.pop(0)
                self.pools.take_snapshot(key, length, seq.slot, seq.table)
            if e == len(seq.prep.ids):
                seq.embeds = None
                self.prefilling.remove(seq)
                drawn.append(seq)
                drawn_rows.append(len(decode) + i)
        t_launch = time.perf_counter()
        # The device runs the step: fill the masks its draws need
        self._fill_masks([s for s in drawn if not s.mask_ready])
        t_masks = time.perf_counter()
        if len(drawn_rows) < len(seqs):
            logits = logits[drawn_rows]
        tokens = self._draw(drawn, logits)
        t_drawn = time.perf_counter()
        self._take(drawn, tokens)
        t_end = time.perf_counter()
        self.counts["steps"] += 1
        self.counts["graph_steps"] += graph
        self.counts["mixed_steps"] += bool(chunks) and bool(decode)
        self.counts["prompt_tokens"] += n_prompt
        self.times["build"] += t_build - t0
        self.times["launch"] += t_launch - t_build
        self.times["grammar"] += t_masks - t_launch
        self.times["sample"] += t_drawn - t_masks
        self.times["accept"] += t_end - t_drawn
        record = None
        if self.log is not None:
            record = {
                "kind": "graph" if graph else "eager",
                "decode": len(decode),
                "decode_tokens": cu[len(decode)],
                "prefill_seqs": len(chunks),
                "prefill_tokens": n_prompt,
                "images": len(rows),
                "build_ms": round((t_build - t0) * 1e3, 3),
                "launch_ms": round((t_launch - t_build) * 1e3, 3),
                "masks_ms": round((t_masks - t_launch) * 1e3, 3),
                "draw_ms": round((t_drawn - t_masks) * 1e3, 3),
                "accept_ms": round((t_end - t_drawn) * 1e3, 3),
                "waiting": len(self.waiting),
                "prefilling": len(self.prefilling),
                "decoding": len(self.decoding),
                "free_pages": len(self.pools.free_pages),
            }
        self.timing.append((start, end, record))
        self._timed()
        return True

    def _timed(self, wait: bool = False) -> None:
        """GPU time of the steps whose end event has completed (all of them with ``wait``); a step
        that draws nothing has no host sync, so its events are read once the GPU passes them."""
        while self.timing and (wait or self.timing[0][1].query()):
            start, end, record = self.timing.popleft()
            end.synchronize()
            gpu_ms = start.elapsed_time(end)
            self.times["gpu"] += gpu_ms / 1e3
            if record is not None:
                self.log.write({**record, "gpu_ms": round(gpu_ms, 3)})

    # ---- drawing -----------------------------------------------------------------------------

    def _fill_masks(self, seqs: list[Seq]) -> None:
        constrained = [s for s in seqs if s.matcher is not None]
        self.grammars.fill(
            [s.matcher for s in constrained], self.masks, [s.slot for s in constrained]
        )
        for s in seqs:
            s.mask_ready = True

    def _draw(self, seqs: list[Seq], logits: torch.Tensor) -> list[int]:
        """One token per row: the grammar mask, then the temperature's draw (the noise
        ``probs / Exp(1)``'s argmax), or the argmax at temperature 0."""
        index = torch.tensor([s.slot for s in seqs], dtype=torch.long)
        masks = self.masks.index_select(0, index)
        free = [i for i, s in enumerate(seqs) if s.matcher is None]
        if free:
            masks[free] = -1
        self.grammars.apply(logits, masks)
        temps = torch.tensor([s.prep.temperature for s in seqs]).to(self.dev, non_blocking=True)
        greedy = logits.argmax(-1)
        probs = torch.softmax(logits / temps.clamp_min(1e-6)[:, None], dim=-1)
        noise = torch.empty_like(probs).exponential_(generator=self.generator)
        drawn = (probs / noise).argmax(-1)
        return torch.where(temps > 0, drawn, greedy).tolist()

    def _take(self, seqs: list[Seq], tokens: list[int]) -> None:
        going = [seq for seq, tok in zip(seqs, tokens) if self._accept(seq, tok)]
        for seq in going:
            seq.mask_ready = False
        exact = [s for s in going if s.matcher is not None and self.s.jump == "exact"]
        self._fill_masks(exact)
        for seq in going:
            if seq.matcher is not None and self.s.jump != "off" and self._jump(seq):
                continue
            if seq not in self.decoding:
                self.decoding.append(seq)

    def _accept(self, seq: Seq, tok: int) -> bool:
        """Takes ``tok`` as the sequence's next token; False when the sequence is done."""
        if tok in self.stops:
            self._done(seq)
            return False
        if seq.matcher is not None and not seq.matcher.accept_token(tok):
            self._fail(seq, ValueError(f"token {tok} rejected by the grammar"))
            return False
        return self._record(seq, tok)

    def _record(self, seq: Seq, tok: int) -> bool:
        """``tok`` (accepted by the grammar) appended; False when the sequence is done."""
        seq.out.append(tok)
        seq.pending.append(tok)
        self.counts["generated_tokens"] += 1
        if len(seq.out) >= seq.prep.max_tokens or (
            seq.matcher is not None and seq.matcher.is_terminated()
        ):
            self._done(seq)
            return False
        return True

    def _jump(self, seq: Seq) -> bool:
        """Accepts what the grammar forces after the last token; True when that finished the
        sequence.

        ``exact``: while the (filled) mask allows a single token, that token; the row stays
        filled for the next draw. ``string``: the text the grammar forces from here
        (``find_jump_forward_string``), tokenized on its own; the tokens a draw would take may
        differ, so the replies may differ from ``exact``'s."""
        match self.s.jump:
            case "exact":
                row = self.masks[seq.slot].numpy()
                for _ in range(self.s.jump_max):
                    tok = self.grammars.only_token(row)
                    if tok is None:
                        return False
                    if tok in self.stops:
                        self._done(seq)
                        return True
                    if not self._accept(seq, tok):
                        return True
                    self.counts["forced_tokens"] += 1
                    self.grammars.fill_one(seq.matcher, self.masks, seq.slot)
            case "string":
                text = seq.matcher.find_jump_forward_string()
                if not text:
                    return False
                for taken, tok in enumerate(self.tokenizer.encode(text, add_special_tokens=False)):
                    if taken >= self.s.jump_max or tok in self.stops:
                        break
                    if not seq.matcher.accept_token(tok):
                        break
                    if not self._record(seq, tok):
                        return True
                    self.counts["forced_tokens"] += 1
        return False

    def _done(self, seq: Seq) -> None:
        if seq in self.decoding:
            self.decoding.remove(seq)
        self._release(seq)
        self.counts["finished"] += 1
        text = self.tokenizer.decode(seq.out, skip_special_tokens=True)
        seq.finish("ok", text)

    def _fail(self, seq: Seq, exc: Exception) -> None:
        if seq in self.decoding:
            self.decoding.remove(seq)
        self._release(seq)
        self.counts["errors"] += 1
        seq.finish("item", exc)
