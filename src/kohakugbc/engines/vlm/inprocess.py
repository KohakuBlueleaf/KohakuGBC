"""The ``vllm`` entry of the ``vlm`` slot: vLLM's engine in the machine's process, behind
Togeari's stream contract.

Each unit is one ``AsyncLLM`` (vLLM runs its engine core in a child process). ``submit`` hands
the request to the unit's event-loop thread and returns; the loop builds the prompt with the
model's chat template (thinking off), decodes the pictures, generates with the request's JSON
schema as structured output, and answers. ``cancel`` aborts the request in vLLM.

Settings: ``model``; ``units`` (1); ``devices`` (per unit, the CUDA devices it uses; several
mean tensor parallel over them); ``gpu_memory_utilization`` (0.5), ``max_model_len`` (16384),
``max_num_seqs`` (64), ``dtype`` (``bfloat16``), ``capacity`` (requests in flight per unit;
default twice ``max_num_seqs``), ``engine_args`` (any other ``AsyncEngineArgs`` field),
``sampling`` (extra ``SamplingParams`` fields).
"""

import asyncio
import io
import itertools
import os
import threading
import uuid

from PIL import Image
from togeari import Cancelled, EngineBase, ItemError, Load, Ok, UnitError
from transformers import AutoProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

from kohakugbc.engines import placement
from kohakugbc.engines.adapters import VllmChatAdapter
from kohakugbc.engines.registry import Slot, entry
from kohakugbc.nodes.specs import chat as specs

_START_LOCK = threading.Lock()


class VllmUnit(EngineBase):
    kind = "stream"
    protocol = "vllm"
    shares_memory = True

    def __init__(
        self,
        model: str,
        devices: list[int] | None = None,
        capacity=None,
        name=None,
        sampling: dict | None = None,
        **engine_args,
    ):
        super().__init__(name or "vllm")
        self.model, self.devices = model, list(devices or [])
        self.engine_args = {
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5,
            "max_model_len": 16384,
            "max_num_seqs": 64,
            "limit_mm_per_prompt": {"image": 4},
            **engine_args,
        }
        if len(self.devices) > 1:
            self.engine_args.setdefault("tensor_parallel_size", len(self.devices))
        self.cap = int(capacity or 2 * self.engine_args["max_num_seqs"])
        self.sampling = dict(sampling or {})
        self.lock = threading.Lock()
        self.inflight = 0
        self.tickets: dict[int, dict] = {}
        self.counter = itertools.count(1)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.llm: AsyncLLM | None = None
        self.processor = None
        self.counts = {"items": 0, "ok": 0, "errors": 0, "cancelled": 0}

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        if self.llm is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.model)
        ready = threading.Event()
        failure: list = []

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.llm = self.loop.run_until_complete(self._create())
            except Exception as exc:  # noqa: BLE001
                failure.append(exc)
                ready.set()
                return
            ready.set()
            self.loop.run_forever()

        # The devices this unit sees, while its engine core starts (it inherits the environment)
        with _START_LOCK:
            saved = os.environ.get("CUDA_VISIBLE_DEVICES")
            if self.devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.devices)
            try:
                threading.Thread(target=run_loop, name=f"{self.name}-loop", daemon=True).start()
                ready.wait()
            finally:
                if saved is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved
        if failure:
            raise failure[0]

    async def _create(self) -> AsyncLLM:
        args = AsyncEngineArgs(model=self.model, **self.engine_args)
        return AsyncLLM.from_engine_args(args)

    def stop(self) -> None:
        if self.llm is None:
            return
        llm, self.llm = self.llm, None
        llm.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)
        with self.lock:
            open_, self.tickets = self.tickets, {}
        for ticket, state in open_.items():
            state["done"](ticket, [UnitError(RuntimeError(f"{self.name} stopped"))])

    def ready(self) -> bool:
        return self.llm is not None

    def spec(self) -> dict:
        return {"engine": "vllm", "model": self.model, "args": dict(self.engine_args)}

    # ---- the contract ------------------------------------------------------------------------

    def capacity(self) -> int:
        with self.lock:
            return max(0, self.cap - self.inflight)

    def load(self) -> Load:
        with self.lock:
            n, seqs = self.inflight, self.engine_args["max_num_seqs"]
            return Load(running=min(n, seqs), waiting=max(0, n - seqs), extra=dict(self.counts))

    def submit(self, work, done):
        ticket = next(self.counter)
        state = {"done": done, "rid": uuid.uuid4().hex, "cancelled": False}
        with self.lock:
            self.inflight += 1
            self.tickets[ticket] = state
            self.counts["items"] += 1
            state["future"] = asyncio.run_coroutine_threadsafe(
                self._run(ticket, work, state), self.loop
            )
        return ticket

    def cancel(self, ticket) -> None:
        """Cancels the request's task: vLLM's ``generate`` aborts the request itself when its
        task is cancelled, wherever the request is (an ``abort`` sent while vLLM is still
        adding the request can leave it running unseen)."""
        with self.lock:
            state = self.tickets.get(ticket)
            if state is None:
                return
            state["cancelled"] = True
            future = state["future"]
        future.cancel()

    # ---- the work, on the unit's loop ------------------------------------------------------------

    def _prompt(self, request: dict) -> dict:
        content = []
        for label, _ in request["images"]:
            if label is not None:
                content.append({"type": "text", "text": label})
            content.append({"type": "image"})
        content.append({"type": "text", "text": request["query"]})
        messages = []
        if request["system"] is not None:
            messages.append({"role": "system", "content": request["system"]})
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        pictures = []
        for _, picture in request["images"]:
            if picture is None:
                raise ValueError("an empty crop cannot be sent")
            with Image.open(io.BytesIO(picture["jpeg"])) as img:
                pictures.append(img.convert("RGB"))
        prompt = {"prompt": text}
        if pictures:
            prompt["multi_modal_data"] = {"image": pictures}
        return prompt

    async def _run(self, ticket: int, item, state: dict) -> None:
        try:
            request = item.payload()
            prompt = await asyncio.get_running_loop().run_in_executor(None, self._prompt, request)
            schema = specs.schema(request["schema"]) if request["schema"] else None
            params = SamplingParams(
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                structured_outputs=StructuredOutputsParams(json=schema) if schema else None,
                **self.sampling,
            )
            final = None
            async for out in self.llm.generate(prompt, params, state["rid"]):
                final = out
            text = final.outputs[0].text if final is not None and final.outputs else ""
            result = Cancelled() if state["cancelled"] else Ok(item.finish(text), True)
        except asyncio.CancelledError:
            result = Cancelled()
        except Exception as exc:  # noqa: BLE001
            result = Cancelled() if state["cancelled"] else ItemError(exc)
        with self.lock:
            if self.tickets.pop(ticket, None) is None:
                return
            self.inflight -= 1
            key = {Ok: "ok", Cancelled: "cancelled"}.get(type(result), "errors")
            self.counts[key] += 1
        state["done"](ticket, [result])


@entry("vlm", "vllm")
def vllm_units(settings: dict, shared: dict) -> Slot:
    settings = dict(settings)
    model = settings.pop("model")
    n = int(settings.pop("units", 1))
    devices = settings.pop("devices", None) or [None] * n
    capacity = settings.pop("capacity", None)
    sampling = settings.pop("sampling", None)
    lane = settings.pop("lane", None) or {}
    extra = settings.pop("engine_args", None) or {}
    units = [
        VllmUnit(
            model,
            devices=d if isinstance(d, list) else ([] if d is None else [d]),
            capacity=capacity,
            name=f"vllm-{i}",
            sampling=sampling,
            **settings,
            **extra,
        )
        for i, d in enumerate(devices[:n])
    ]

    # Each unit reserves its share of every device it uses (vLLM's gpu_memory_utilization)
    needs = []
    for unit in units:
        share = float(unit.engine_args["gpu_memory_utilization"])
        for d in unit.devices or [0]:
            mem = placement.memory(f"cuda:{d}")
            if mem is not None:
                needs.append(
                    placement.Need(unit.name, f"cuda:{d}", int(share * mem[1]), "reservation")
                )
    return Slot(units=units, adapter=VllmChatAdapter(), lane=dict(lane), needs=needs)
