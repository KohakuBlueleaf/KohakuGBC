"""The ``vllm`` entry of the ``vlm`` slot: vLLM's engine in the machine's process, behind
Togeari's stream contract.

Each unit is one ``AsyncLLM`` (vLLM runs its engine core in a child process). ``submit`` hands
the request to the unit's event-loop thread and returns; the loop builds the prompt with the
model's chat template (thinking off), decodes the pictures, generates, and answers. ``cancel``
cancels the request's task, and vLLM's ``generate`` aborts the request itself when its task is
cancelled, wherever the request is (an ``abort`` sent while vLLM is still adding the request can
leave it running unseen).

``structured`` picks how a request's JSON schema holds the reply: ``prompt`` (the prompts already
describe the JSON; a reply that GBC's reader cannot read as the schema's object is asked again, up
to ``retries`` times, and the last reply is answered either way), ``grammar`` (vLLM's structured
outputs constrain decoding to the schema) or ``kohaku`` (KohakuvLLM's grammar logits processor,
``kohakuvllm.grammar``, constrains it; the engine gets the processor from this setting). On
Qwen3.5-4B with GBC's requests, 1 reply in 400 needs the retry, and ``grammar`` (xgrammar) costs
40% of the machine's throughput.

Settings: ``model``; ``units`` (1); ``devices`` (per unit, the CUDA devices it uses; several
mean tensor parallel over them); ``gpu_memory_utilization`` (0.5), ``max_model_len`` (16384),
``max_num_seqs`` (64), ``dtype`` (``bfloat16``), ``capacity`` (requests in flight per unit;
default twice ``max_num_seqs``), ``structured`` (``prompt``), ``retries`` (1), ``engine_args``
(any other ``AsyncEngineArgs`` field: ``mamba_ssm_cache_dtype``, ``scheduler_cls``, ...),
``sampling`` (extra ``SamplingParams`` fields), ``stats_s`` (0: off; else vLLM's stats line
every that many seconds, with ``disable_log_stats=False`` in ``engine_args``); ``frontend``
(``vllm``: vLLM's own input processing and image encoder; ``kohakuvllm``: KohakuvLLM's frontend,
requests prepared in worker processes and images encoded by its vision tower in this process,
handed to the engine as token ids and image embeddings), ``frontend_args`` (``workers``,
``vision_fp8``, ``max_patches``) and ``frontend_gb`` (1.5: the vision tower's reservation).
"""

import asyncio
import io
import itertools
import json
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
from kohakugbc.nodes.specs.parse import reply_ok

try:
    from kohakuvllm.front import engine as kohaku_front
except ImportError:
    kohaku_front = None

_START_LOCK = threading.Lock()
STRUCTURED = ("prompt", "grammar", "kohaku")
FRONTENDS = ("vllm", "kohakuvllm")
KOHAKU_GRAMMAR = "kohakuvllm.grammar:JsonSchemaLogits"


def messages_of(request: dict) -> tuple[list[dict], list[bytes]]:
    """The request's chat messages (an ``{"type": "image"}`` item per picture) and its
    pictures' JPEG bytes."""
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
    jpegs = []
    for _, picture in request["images"]:
        if picture is None:
            raise ValueError("an empty crop cannot be sent")
        jpegs.append(picture["jpeg"])
    return messages, jpegs


def prompt_parts(processor, request: dict) -> tuple[str, list]:
    """The chat-templated prompt text (thinking off) and the request's pictures, decoded."""
    messages, jpegs = messages_of(request)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    pictures = []
    for data in jpegs:
        with Image.open(io.BytesIO(data)) as img:
            pictures.append(img.convert("RGB"))
    return text, pictures


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
        structured: str = "prompt",
        retries: int = 1,
        stats_s: float = 0.0,
        frontend: str = "vllm",
        frontend_args: dict | None = None,
        frontend_gb: float = 1.5,
        **engine_args,
    ):
        super().__init__(name or "vllm")
        if structured not in STRUCTURED:
            raise ValueError(f"structured {structured!r}: one of {STRUCTURED}")
        if frontend not in FRONTENDS:
            raise ValueError(f"frontend {frontend!r}: one of {FRONTENDS}")
        if frontend == "kohakuvllm" and kohaku_front is None:
            raise ValueError("frontend 'kohakuvllm' needs the kohakuvllm package installed")
        self.structured, self.retries, self.stats_s = structured, int(retries), float(stats_s)
        self.frontend, self.frontend_args = frontend, dict(frontend_args or {})
        self.frontend_gb = float(frontend_gb) if frontend == "kohakuvllm" else 0.0
        self.front = None
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
        if frontend == "kohakuvllm":
            self.engine_args = kohaku_front.engine_args(self.engine_args)
        if structured == "kohaku":
            procs = list(self.engine_args.get("logits_processors") or [])
            if KOHAKU_GRAMMAR not in procs:
                self.engine_args["logits_processors"] = [*procs, KOHAKU_GRAMMAR]
        self.cap = int(capacity or 2 * self.engine_args["max_num_seqs"])
        self.sampling = dict(sampling or {})
        self.lock = threading.Lock()
        self.inflight = 0
        self.tickets: dict[int, dict] = {}
        self.counter = itertools.count(1)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.llm: AsyncLLM | None = None
        self.processor = None
        self.counts = {"items": 0, "ok": 0, "errors": 0, "cancelled": 0, "retries": 0}

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
        # Built after the devices are restored: its vision tower runs in this process
        if self.frontend == "kohakuvllm":
            device = f"cuda:{self.devices[0]}" if self.devices else "cuda:0"
            self.front = kohaku_front.Frontend(
                self.llm, self.model, vision_device=device, **self.frontend_args
            )

    async def _create(self) -> AsyncLLM:
        args = AsyncEngineArgs(model=self.model, **self.engine_args)
        llm = AsyncLLM.from_engine_args(args)
        if self.stats_s > 0:
            asyncio.get_running_loop().create_task(self._log_stats(llm))
        return llm

    async def _log_stats(self, llm: AsyncLLM) -> None:
        """vLLM's own stats log line (running / waiting requests, throughput, cache use) every
        ``stats_s`` seconds; vLLM prints it only when asked."""
        while self.llm is None or self.llm is llm:
            await asyncio.sleep(self.stats_s)
            await llm.do_log_stats()

    def stop(self) -> None:
        if self.llm is None:
            return
        llm, self.llm = self.llm, None
        if self.front is not None:
            self.front.close()
            self.front = None
        llm.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)
        with self.lock:
            open_, self.tickets = self.tickets, {}
        for ticket, state in open_.items():
            state["done"](ticket, [UnitError(RuntimeError(f"{self.name} stopped"))])

    def ready(self) -> bool:
        return self.llm is not None

    def spec(self) -> dict:
        return {
            "engine": "vllm",
            "model": self.model,
            "args": dict(self.engine_args),
            "frontend": self.frontend,
            "frontend_args": dict(self.frontend_args),
        }

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
        with self.lock:
            state = self.tickets.get(ticket)
            if state is None:
                return
            state["cancelled"] = True
            future = state["future"]
        future.cancel()

    # ---- the work, on the unit's loop ------------------------------------------------------------

    async def _generate(self, prompt: dict, params: SamplingParams, rid: str) -> str:
        final = None
        async for out in self.llm.generate(prompt, params, rid):
            final = out
        return final.outputs[0].text if final is not None and final.outputs else ""

    async def _run(self, ticket: int, item, state: dict) -> None:
        try:
            request = item.payload()
            if self.front is not None:
                prompt = await self.front.inputs(*messages_of(request))
            else:
                text, pictures = await asyncio.get_running_loop().run_in_executor(
                    None, prompt_parts, self.processor, request
                )
                prompt = {"prompt": text}
                if pictures:
                    prompt["multi_modal_data"] = {"image": pictures}
            schema = specs.schema(request["schema"]) if request["schema"] else None
            grammar = schema is not None and self.structured != "prompt"
            vllm_grammar = grammar and self.structured == "grammar"
            kohaku = grammar and self.structured == "kohaku"
            params = SamplingParams(
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                structured_outputs=StructuredOutputsParams(json=schema) if vllm_grammar else None,
                extra_args={"kohaku_json_schema": json.dumps(schema)} if kohaku else None,
                **self.sampling,
            )
            for attempt in range(self.retries + 1):
                rid = state["rid"] if attempt == 0 else f"{state['rid']}-{attempt}"
                reply = await self._generate(prompt, params, rid)
                if grammar or reply_ok(reply, request["schema"], schema):
                    break
                if attempt < self.retries:
                    with self.lock:
                        self.counts["retries"] += 1
            result = Cancelled() if state["cancelled"] else Ok(item.finish(reply), True)
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
    lane = settings.pop("lane", None) or {}
    extra = settings.pop("engine_args", None) or {}
    units = [
        VllmUnit(
            model,
            devices=d if isinstance(d, list) else ([] if d is None else [d]),
            name=f"vllm-{i}",
            **settings,
            **extra,
        )
        for i, d in enumerate(devices[:n])
    ]

    # Each unit reserves its share of every device it uses (vLLM's gpu_memory_utilization), and
    # KohakuvLLM's vision tower on its first device
    needs = []
    for unit in units:
        share = float(unit.engine_args["gpu_memory_utilization"])
        for d in unit.devices or [0]:
            mem = placement.memory(f"cuda:{d}")
            if mem is not None:
                needs.append(
                    placement.Need(unit.name, f"cuda:{d}", int(share * mem[1]), "reservation")
                )
        if unit.frontend_gb:
            device = f"cuda:{(unit.devices or [0])[0]}"
            gb = int(unit.frontend_gb * placement.GIB)
            needs.append(placement.Need(f"{unit.name}:vision", device, gb, "reservation"))
    return Slot(units=units, adapter=VllmChatAdapter(), lane=dict(lane), needs=needs)
