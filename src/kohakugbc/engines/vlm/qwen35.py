"""The ``qwen35`` entry of KohakuGBC's ``vlm`` slot: this engine behind Togeari's stream contract.

``submit`` hands the request to a pool of preparing threads (chat template, image processing,
positions) and returns; prepared requests with images pass the vision stage
(``qwen35.serve.vision``), then the engine thread (``qwen35.serve.engine``) answers each with
the reply text. ``cancel`` drops the request wherever it is.

Settings: ``model`` (a Qwen3.5 checkpoint folder or Hugging Face name); ``device``
(``cuda:0``); ``memory`` (0.5: the share of the device's memory the unit holds, weights
included); ``max_seqs`` (64);
``max_model_len`` (16384); ``page_size`` (16); ``prefill_tokens`` (4096 prompt tokens per
step), ``prefill_min`` (512) and ``prefill_wait_ms`` (50: when prefill joins a step);
``snapshots`` (16) and ``boundaries`` (``["system"]``; also ``"images"``); ``state_dtype``
(``float32``, or ``bfloat16``); ``jump`` (``exact``, ``string`` or ``off``: what the grammar
forces is taken without a forward, see ``Engine._jump``) and ``jump_max`` (16 tokens at a
time); ``any_whitespace`` (True, as vLLM compiles schemas: whitespace is allowed between any
two tokens, so ``exact`` finds no forced token; False keeps the JSON compact); ``cuda_graphs``
(True); ``vision_patches`` (32768 patches per vision call); ``prefer`` (op backends in order,
e.g. ``{"delta": ["fla", "triton"]}``) and ``race`` (True); ``prep_workers`` (4);
``capacity`` (requests in flight; twice ``max_seqs``); ``seed`` (0); ``workspace_gib``
(activation room; estimated from ``prefill_tokens`` by default); ``step_log`` (a path: one
JSON line per step and vision call).
"""

import gc
import itertools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import torch
from togeari import Cancelled, EngineBase, ItemError, Load, Ok, UnitError
from transformers import AutoProcessor

from kohakugbc.engines import placement
from kohakugbc.engines.adapters import VllmChatAdapter
from kohakugbc.engines.registry import Slot, entry
from kohakugbc.nodes.specs import chat as specs
from qwen35.model.forward import Model
from qwen35.ops import Ops
from qwen35.ops import backends as _backends  # noqa: F401  (registers every backend)
from qwen35.serve.engine import Engine, Seq, Settings
from qwen35.serve.grammar import Grammars
from qwen35.serve.graphs import DecodeGraphs
from qwen35.serve.inputs import Preparer
from qwen35.serve.pools import Pools, plan
from qwen35.serve.steplog import StepLog
from qwen35.serve.vision import VisionStage

GIB = 1024**3
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PREFER = {
    "delta": ["triton", "fla"],
    "conv": ["torch"],
    "attention": ["triton", "sdpa"],
}


def workspace_bytes(shape, prefill_tokens: int, max_seqs: int, vision_patches: int) -> int:
    """Activation room of a step of ``prefill_tokens`` (the MLP, the convolution's fp32 sums
    and the hidden states per token), of the sampling of ``max_seqs`` rows (three fp32 copies of
    the logits), of a vision call of ``vision_patches`` patches (40 KiB each, at the same time),
    and 512 MiB for the kernels."""
    per_token = 2 * 3 * shape.intermediate + 4 * 3 * shape.conv_channels + 2 * 12 * shape.hidden
    logits = 3 * 4 * shape.vocab * max_seqs
    return prefill_tokens * per_token + logits + vision_patches * 40 * 1024 + GIB // 2


class Qwen35Unit(EngineBase):
    kind = "stream"
    protocol = "request"
    shares_memory = True

    def __init__(
        self,
        model: str,
        name: str | None = None,
        device: str = "cuda:0",
        memory: float = 0.5,
        max_seqs: int = 64,
        max_model_len: int = 16384,
        page_size: int = 16,
        prefill_tokens: int = 4096,
        prefill_min: int = 512,
        prefill_wait_ms: float = 50.0,
        snapshots: int = 16,
        boundaries=("system",),
        state_dtype: str = "float32",
        jump: str = "exact",
        jump_max: int = 16,
        any_whitespace: bool = True,
        cuda_graphs: bool = True,
        vision_patches: int = 32768,
        prefer: dict | None = None,
        race: bool = True,
        prep_workers: int = 4,
        capacity: int | None = None,
        seed: int = 0,
        workspace_gib: float | None = None,
        step_log: str | None = None,
    ):
        super().__init__(name or "qwen35")
        if state_dtype not in DTYPES:
            raise ValueError(f"state_dtype {state_dtype!r}: one of {sorted(DTYPES)}")
        if jump not in ("exact", "string", "off"):
            raise ValueError(f"jump {jump!r}: one of exact, string, off")
        self.model_path, self.device = model, device
        self.options = {
            "memory": float(memory),
            "max_seqs": int(max_seqs),
            "max_model_len": int(max_model_len),
            "page_size": int(page_size),
            "prefill_tokens": int(prefill_tokens),
            "prefill_min": int(prefill_min),
            "prefill_wait_ms": float(prefill_wait_ms),
            "snapshots": int(snapshots),
            "boundaries": list(boundaries),
            "state_dtype": state_dtype,
            "jump": jump,
            "jump_max": int(jump_max),
            "any_whitespace": bool(any_whitespace),
            "cuda_graphs": bool(cuda_graphs),
            "vision_patches": int(vision_patches),
            "prefer": {**PREFER, **(prefer or {})},
            "race": bool(race),
            "prep_workers": int(prep_workers),
            "seed": int(seed),
            "workspace_gib": workspace_gib,
            "step_log": step_log,
        }
        self.cap = int(capacity or 2 * max_seqs)
        self.lock = threading.Lock()
        self.inflight = 0
        self.tickets: dict[int, dict] = {}
        self.counter = itertools.count(1)
        self.engine: Engine | None = None
        self.model = self.ops = self.pools = self.graphs = None
        self.prep: ThreadPoolExecutor | None = None
        self.preparer = None
        self.vision: VisionStage | None = None
        self.log: StepLog | None = None
        self.plan = None
        self.last_stats: dict = {}
        self.prep_s = 0.0

    # ---- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if self.engine is not None:
            return
        try:
            self._start()
        except BaseException:
            self.engine = self.graphs = self.pools = self.ops = self.model = self.preparer = None
            self.vision = None
            gc.collect()
            torch.cuda.empty_cache()
            raise

    def _start(self) -> None:
        o = self.options
        dev = torch.device(self.device)
        torch.cuda.set_device(dev)
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(dev)
        # What this process's allocator holds unused is free to it too
        free += torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        before = torch.cuda.memory_allocated(dev)
        self.model = Model(self.model_path, self.device)
        shape = self.model.shape
        processor = AutoProcessor.from_pretrained(self.model.path)
        tok = processor.tokenizer
        stop = {shape.eos, tok.convert_tokens_to_ids("<|im_end|>")}
        weights = torch.cuda.memory_allocated(dev) - before
        work = (
            int(o["workspace_gib"] * GIB)
            if o["workspace_gib"] is not None
            else workspace_bytes(shape, o["prefill_tokens"], o["max_seqs"], o["vision_patches"])
        )
        budget = min(int(o["memory"] * total) - weights, free - weights) - work
        state_dtype = DTYPES[o["state_dtype"]]
        self.plan = plan(
            shape,
            budget,
            o["max_seqs"],
            o["snapshots"],
            o["page_size"],
            state_dtype,
            self.model.dtype,
            o["max_model_len"],
        )
        if self.plan.reasons:
            raise ValueError(f"{self.name}: " + "; ".join(self.plan.reasons))
        self.pools = Pools(shape, self.plan, dev, state_dtype, self.model.dtype, o["snapshots"])
        self.ops = Ops(o["prefer"], race=o["race"])
        grammars = Grammars(tok, shape.vocab, sorted(stop), any_whitespace=o["any_whitespace"])
        max_pages = -(-o["max_model_len"] // o["page_size"])
        settings = Settings(
            max_seqs=o["max_seqs"],
            prefill_tokens=o["prefill_tokens"],
            prefill_min=o["prefill_min"],
            prefill_wait_ms=o["prefill_wait_ms"],
            jump=o["jump"],
            jump_max=o["jump_max"],
            seed=o["seed"],
        )
        self.log = StepLog(o["step_log"]) if o["step_log"] else None
        self.engine = Engine(
            self.model, self.ops, self.pools, grammars, tok, settings, max_pages, stop, self.log
        )
        if o["cuda_graphs"]:
            self.graphs = DecodeGraphs(self.model, self.ops, self.pools, o["max_seqs"], max_pages)
            with torch.inference_mode():
                self.graphs.capture()
        self.vision = VisionStage(
            self.model, self.pools, self.engine.add, o["vision_patches"], self.log
        )
        self.preparer = Preparer(
            processor,
            self.model.hf.model.get_rope_index,
            shape,
            specs.schema,
            boundaries=o["boundaries"],
        )
        self.prep = ThreadPoolExecutor(o["prep_workers"], thread_name_prefix=f"{self.name}-prep")
        self.engine.start(self.graphs)
        self.vision.start()

    def stop(self) -> None:
        if self.engine is None:
            return
        self.vision.stop()
        self.engine.stop()
        self.last_stats = self.stats()
        self.engine = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.prep.shutdown(wait=False, cancel_futures=True)
        with self.lock:
            open_, self.tickets = self.tickets, {}
            self.inflight = 0
        for ticket, state in open_.items():
            state["done"](ticket, [UnitError(RuntimeError(f"{self.name} stopped"))])
        self.graphs = self.pools = self.ops = self.model = self.preparer = self.prep = None
        self.vision = None
        gc.collect()
        torch.cuda.empty_cache()

    def ready(self) -> bool:
        return (
            self.engine is not None and self.engine.failure is None and self.vision.failure is None
        )

    def spec(self) -> dict:
        out = {"engine": "qwen35", "model": self.model_path, "device": self.device, **self.options}
        if self.plan is not None:
            out["plan"] = {k: v for k, v in asdict(self.plan).items() if k != "reasons"}
        return out

    def stats(self) -> dict:
        """Counters, times and the op choices bound so far (after ``stop``: at the stop)."""
        e = self.engine
        if e is None:
            return dict(self.last_stats)
        return {
            "counts": {**e.counts, **self.pools.counts, **self.vision.counts},
            "times_s": {**e.times, **self.vision.times, "prepare": self.prep_s},
            "ops": {f"{op}[{key}]": c for (op, key), c in self.ops.choices.items()},
            "failures": {
                stage: part.failure_text
                for stage, part in (("engine", e), ("vision", self.vision))
                if part.failure_text
            },
        }

    # ---- the contract ------------------------------------------------------------------------

    def capacity(self) -> int:
        with self.lock:
            return max(0, self.cap - self.inflight)

    def load(self) -> Load:
        running, waiting = self.engine.load() if self.engine is not None else (0, 0)
        with self.lock:
            preparing = max(0, self.inflight - running - waiting)
        extra = dict(self.engine.counts) if self.engine is not None else {}
        return Load(running=running, waiting=waiting + preparing, extra=extra)

    def submit(self, work, done):
        ticket = next(self.counter)
        state = {"done": done, "item": work, "cancelled": False}
        with self.lock:
            self.inflight += 1
            self.tickets[ticket] = state
        self.prep.submit(self._prepare, ticket, work, state)
        return ticket

    def cancel(self, ticket) -> None:
        with self.lock:
            state = self.tickets.get(ticket)
            if state is None:
                return
            state["cancelled"] = True
        if self.engine is not None:
            self.engine.cancel(ticket)

    def _prepare(self, ticket: int, item, state: dict) -> None:
        if state["cancelled"]:
            self._answer(ticket, Cancelled())
            return
        t0 = time.perf_counter()
        try:
            prep = self.preparer(item.payload())
        except Exception as exc:  # noqa: BLE001
            # The answer outlives this call: its frames' locals (the preparer, the model) must not
            traceback.clear_frames(exc.__traceback__)
            self._answer(ticket, ItemError(exc))
            return
        with self.lock:
            self.prep_s += time.perf_counter() - t0
        engine, vision = self.engine, self.vision
        if engine is None:
            return

        def finish(kind, value, ticket=ticket, item=item):
            match kind:
                case "ok":
                    result = Ok(item.finish(value), True)
                case "cancelled":
                    result = Cancelled()
                case "item":
                    result = ItemError(value)
                case _:
                    result = UnitError(value)
            self._answer(ticket, result)

        seq = Seq(ticket=ticket, prep=prep, finish=finish, state=state)
        if prep.pixels is not None:
            vision.add(seq)
        else:
            engine.add(seq)

    def _answer(self, ticket: int, result) -> None:
        with self.lock:
            state = self.tickets.pop(ticket, None)
            if state is None:
                return
            self.inflight -= 1
        state["done"](ticket, [result])


@entry("vlm", "qwen35")
def qwen35_units(settings: dict, shared: dict) -> Slot:
    settings = dict(settings)
    model = settings.pop("model")
    lane = settings.pop("lane", None) or {}
    devices = settings.pop("devices", None) or [settings.pop("device", "cuda:0")]
    units = [
        Qwen35Unit(
            model, name=f"qwen35-{i}", device=f"cuda:{d}" if isinstance(d, int) else d, **settings
        )
        for i, d in enumerate(devices)
    ]
    needs = []
    for unit in units:
        mem = placement.memory(unit.device)
        if mem is not None:
            share = unit.options["memory"]
            needs.append(placement.Need(unit.name, unit.device, int(share * mem[1]), "reservation"))
    return Slot(units=units, adapter=VllmChatAdapter(), lane=dict(lane), needs=needs)
