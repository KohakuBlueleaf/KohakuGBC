"""The ``tokenomos`` entry of the ``vlm`` slot: Tokenomos's in-process ``LLM`` behind Togeari's
stream contract, in an engine host only.

The entry runs inside a ``gbc host`` and the machine reaches it through the ``remote`` entry, so
requests cross the host's connector (pictures through shared memory on one node) and the engine
shares no GIL with the machine.

``submit`` hands the request to a small thread pool and returns; the pool reads the payload and
submits it to the ``LLM``, whose prep workers render, decode and tokenize. A reply ends on the
engine's thread. A failed engine ends the host: a device-side error kills the process's CUDA
context for every engine in it, so the machine's remote units answer ``UnitError`` for their open
work and start the host again.

Settings: ``model``, ``device`` (``cuda``; the host's environment picks the card), ``arch``
(kwargs of the model declaration, e.g. ``{"head": "mxfp8"}``), ``engine`` (kwargs of ``Engine``:
``slots``, ``max_len``, ``reserve_new``, ...), ``leave_gb`` (0: memory kept for other engines on
the card), ``prep_workers`` (16), ``capacity`` (requests in flight, default four times the
slots), ``structured`` (``grammar``: the device grammar holds the reply to the request's schema;
``prompt``: the prompts ask for the JSON, a reply GBC's reader cannot read is asked again up to
``retries`` times), ``retries`` (1), ``sampling`` (extra ``SamplingParams`` fields),
``chat_kwargs`` (the chat template's switches, ``{"enable_thinking": False}`` under any given),
``memory_gb`` (the plan's reservation on the device, default 90% of it less ``leave_gb``).
"""

import itertools
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from togeari import Cancelled, EngineBase, ItemError, Load, Ok, UnitError
from tokenomos import LLM, SamplingParams, models

from kohakugbc.engines import placement
from kohakugbc.engines.adapters import VllmChatAdapter
from kohakugbc.engines.registry import Slot, entry
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs.parse import reply_ok

STRUCTURED = ("grammar", "prompt")
GIB = 1 << 30


def messages_of(request: dict) -> list[dict]:
    """The request's chat messages, each picture's JPEG bytes in its image item."""
    content = []
    for label, picture in request["images"]:
        if label is not None:
            content.append({"type": "text", "text": label})
        if picture is None:
            raise ValueError("an empty crop cannot be sent")
        content.append({"type": "image", "image": picture["jpeg"]})
    content.append({"type": "text", "text": request["query"]})
    messages = []
    if request["system"] is not None:
        messages.append({"role": "system", "content": request["system"]})
    messages.append({"role": "user", "content": content})
    return messages


class TokenomosUnit(EngineBase):
    kind = "stream"
    protocol = "request"
    shares_memory = True

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        arch: dict | None = None,
        engine: dict | None = None,
        leave_gb: float = 0.0,
        prep_workers: int = 16,
        capacity: int | None = None,
        structured: str = "grammar",
        retries: int = 1,
        sampling: dict | None = None,
        chat_kwargs: dict | None = None,
        name: str | None = None,
    ):
        super().__init__(name or "tokenomos")
        if structured not in STRUCTURED:
            raise ValueError(f"structured {structured!r}: one of {STRUCTURED}")
        self.model, self.device = model, device
        self.arch, self.engine_kw = dict(arch or {}), dict(engine or {})
        self.engine_kw.setdefault("kv_leave", int(leave_gb * GIB))
        self.prep_workers, self.cap = prep_workers, capacity
        self.structured, self.retries = structured, int(retries)
        self.sampling = dict(sampling or {})
        self.chat_kwargs = {"enable_thinking": False, **(chat_kwargs or {})}
        self.lock = threading.Lock()
        self.inflight = 0
        self.tickets: dict[int, dict] = {}
        self.counter = itertools.count(1)
        self.llm: LLM | None = None
        self.front: ThreadPoolExecutor | None = None
        self.watch: threading.Thread | None = None
        self.counts = {"items": 0, "ok": 0, "errors": 0, "cancelled": 0, "retries": 0}

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        if self.llm is not None:
            return
        arch = models.configure(self.model, **self.arch)
        self.llm = LLM(
            self.model,
            device=self.device,
            arch=arch,
            prep_workers=self.prep_workers,
            **self.engine_kw,
        )
        self.cap = int(self.cap or 4 * self.llm.engine.slots)
        self.front = ThreadPoolExecutor(4, thread_name_prefix=f"{self.name}-front")
        self.watch = threading.Thread(target=self._watch, name=f"{self.name}-watch", daemon=True)
        self.watch.start()

    def _watch(self) -> None:
        """A failed engine ends the process (see the module docstring)."""
        llm = self.llm
        while self.llm is llm:
            if llm.failure is not None:
                traceback.print_exception(llm.failure)
                os._exit(1)
            threading.Event().wait(0.5)

    def _fail_all(self, result) -> None:
        with self.lock:
            open_, self.tickets = self.tickets, {}
            self.inflight = 0
        for ticket, state in open_.items():
            state["done"](ticket, [result])

    def stop(self) -> None:
        if self.llm is None:
            return
        llm, self.llm = self.llm, None
        self.front.shutdown(wait=False, cancel_futures=True)
        llm.close()
        self._fail_all(UnitError(RuntimeError(f"{self.name} stopped")))

    def ready(self) -> bool:
        return self.llm is not None and self.llm.failure is None

    def spec(self) -> dict:
        return {
            "engine": "tokenomos",
            "model": self.model,
            "arch": dict(self.arch),
            "args": dict(self.engine_kw),
            "structured": self.structured,
        }

    # ---- the contract ------------------------------------------------------------------------

    def capacity(self) -> int:
        with self.lock:
            return max(0, (self.cap or 0) - self.inflight) if self.ready() else 0

    def load(self) -> Load:
        slots = self.llm.engine.slots if self.llm is not None else 0
        with self.lock:
            n = self.inflight
            return Load(running=min(n, slots), waiting=max(0, n - slots), extra=dict(self.counts))

    def submit(self, work, done):
        ticket = next(self.counter)
        state = {"done": done, "item": work, "cancelled": False, "attempt": 0}
        with self.lock:
            self.inflight += 1
            self.tickets[ticket] = state
            self.counts["items"] += 1
        self.front.submit(self._first, ticket, state)
        return ticket

    def cancel(self, ticket) -> None:
        """The request runs on; its answer becomes ``Cancelled``."""
        with self.lock:
            state = self.tickets.get(ticket)
            if state is not None:
                state["cancelled"] = True

    # ---- the work ------------------------------------------------------------------------------

    def _first(self, ticket: int, state: dict) -> None:
        try:
            request = state["item"].payload()
            schema = specs.schema(request["schema"]) if request["schema"] else None
            grammar = schema if self.structured == "grammar" else None
            state["params"] = SamplingParams(
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
                json_schema=grammar,
                **self.sampling,
            )
            state["request"], state["schema"] = request, schema
            state["messages"] = messages_of(request)
            self._send(ticket, state)
        except Exception as exc:  # noqa: BLE001
            self._answer(ticket, state, ItemError(exc))

    def _send(self, ticket: int, state: dict) -> None:
        fut = self.llm.submit_chat(state["messages"], state["params"], **self.chat_kwargs)
        fut.add_done_callback(lambda f: self._replied(ticket, state, f))

    def _replied(self, ticket: int, state: dict, fut) -> None:
        try:
            reply = self.llm.output(fut.result()).text
            if (
                self.structured == "prompt"
                and not state["cancelled"]
                and state["attempt"] < self.retries
                and not reply_ok(reply, state["request"]["schema"], state["schema"])
            ):
                state["attempt"] += 1
                with self.lock:
                    self.counts["retries"] += 1
                self._send(ticket, state)
                return
            self._answer(ticket, state, Ok(state["item"].finish(reply), True))
        except Exception as exc:  # noqa: BLE001
            self._answer(ticket, state, ItemError(exc))

    def _answer(self, ticket: int, state: dict, result) -> None:
        result = Cancelled() if state["cancelled"] else result
        with self.lock:
            if self.tickets.pop(ticket, None) is None:
                return
            self.inflight -= 1
            key = {Ok: "ok", Cancelled: "cancelled"}.get(type(result), "errors")
            self.counts[key] += 1
        state["done"](ticket, [result])


@entry("vlm", "tokenomos", hosted=True)
def tokenomos_units(settings: dict, shared: dict) -> Slot:
    settings = dict(settings)
    model = settings.pop("model")
    lane = settings.pop("lane", None) or {}
    memory_gb = settings.pop("memory_gb", None)
    unit = TokenomosUnit(model, name="tokenomos-0", **settings)
    needs = []
    mem = placement.memory(unit.device)
    if mem is not None:
        held = int(memory_gb * GIB) if memory_gb is not None else int(0.9 * mem[1])
        held -= unit.engine_kw["kv_leave"]
        needs.append(placement.Need(unit.name, unit.device, held, "reservation"))
    return Slot(units=[unit], adapter=VllmChatAdapter(), lane=dict(lane), needs=needs)
