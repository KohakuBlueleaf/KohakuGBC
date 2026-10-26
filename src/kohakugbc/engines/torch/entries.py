"""The ``torch`` entries of the ``detect`` and ``embed`` slots: GBC's vision models on Togeari's
torch engine.

Settings: ``model`` (a checkpoint), ``devices`` (``"auto"``, a device, or a list: one unit per
device, a list inside the list one model split across those devices), ``dtype`` (float32 by
default: OWLv2 in float16 gives batched scores up to 0.085 from single ones, float32 within
1.3e-4; float16 with ``max_batch`` 1 is the PoC's own run), ``max_batch``, ``wait_ms`` (the batch
former's wait), ``former`` (``size``: up to ``max_batch`` requests; ``budget``: Togeari's
``BudgetFormer`` over each request's costs, ``budgets`` overriding the defaults: the detector's
``crops`` (1 per request) and ``pairs`` (its texts) at SAM 3's ``max_crops`` / ``max_pairs``, or
``max_batch`` / ``max_batch * max_texts`` for the other detectors; the embedder's ``texts`` and
``chars`` at ``max_batch`` / 4 x ``graph_tokens``; a request over a budget runs alone),
``memory_gb`` (the plan's estimate per device), ``priority`` (the model's priority on its
engine: under ``engine.schedule`` ``priority_streams`` its CUDA stream's, higher first, ahead of
other work on the card such as the VLM's default streams), ``engine`` (``schedule``,
``isolation``, ``slots``, ``memory_cap_gb``, ``prep_workers``; slots whose models share
devices, dtype and these settings share one engine). Detector: ``model`` names OWLv2
(``google/owlv2-base-patch16-ensemble``, the PoC's) or SAM 3 (``facebook/sam3``, or a
SAM3-LiteText checkpoint ``vil-uob/sam3-litetext-*``; ``box_source``: ``pred``, ``mask`` or
``mask_full``, ``mask_threshold`` 0.5, ``retry_singular`` true: a text no box answers is asked
again in its singular form); ``box_threshold`` (0.05), ``max_texts`` (16). SAM 3's
``impl``: ``auto`` (GBC's own model for ``sam3`` / ``sam3_video`` checkpoints, transformers'
otherwise), ``native`` (``max_pairs`` 16 pairs per fusion pass, ``max_crops`` 4 crops per vision
pass, at most ``max_batch``, ``text_cache`` 4096, ``kernels`` true, ``fusion_graph`` true,
``fusion_buckets`` ``exact``, ``text_graph`` true, ``vision_graph`` (the ViT replayed from one
CUDA graph per crop count; false by default), ``fp8`` (``row`` / ``mx`` / ``nvfp4``: the ViT's
linears in fp8 / NVFP4; ``auto``: the fastest the card runs, mx on sm100+, row on sm89, none
below; none by default), ``fp8_mlp`` (the MLPs' recipe, ``fp8``'s by default: ``mx`` with
``nvfp4`` keeps attention in MX), ``fp4_scales`` (NVFP4: a calibrated amax JSON, one producer
pass per input; ``scripts/bench/detector/fp4_calibrate.py`` writes it),
``fp8_encoder`` (the DETR encoder's MLPs too), ``text_device`` (``cpu``: the text encoder on the
host), ``budget_gb`` (caps fitted from the card's measured memory plan), ``capture`` (``load``:
every CUDA graph captured at load, other shapes uncaptured; ``lazy``: on first use);
dtype ``auto``: bfloat16 on sm80+, float16 before) or ``transformers`` (``max_units`` 4 pairs per
decoder pass). Embedder's ``impl``: ``auto`` (GBC's own EmbeddingGemma for ``gemma3_text``
checkpoints, sentence-transformers otherwise), ``native`` (``kernels`` true, ``graphs`` true,
``residual`` ``float32``, ``capture`` ``load`` with ``graph_tokens`` 4096 the largest stream
captured; dtype float32) or ``st``. Tagger (the ``tag`` slot): ``model``
(``pixai-labs/pixai-tagger-v1.0``), ``max_batch`` (4 images per pass), ``fp8`` (as SAM 3's),
``kernels`` true, ``vision_graph`` true (one CUDA graph per batch size, captured at load); dtype
``auto``.
"""

from togeari.engines.torch import TorchAdapter, TorchEngine
from togeari.policies import BudgetFormer, SizeOrWait
from transformers import AutoConfig

from kohakugbc.engines import placement
from kohakugbc.engines.registry import Slot, entry
from kohakugbc.engines.torch.modules import (
    NativeEmbedder,
    NativeSam3Detector,
    NativeTagger,
    Owlv2Detector,
    Sam3Detector,
    SentenceEmbedder,
)
from kohakugbc_models.detect.sam3 import memory
from kohakugbc_models.detect.sam3.model import fp8_recipe

# Checkpoint model types GBC's own models load (``impl: auto`` picks them for these)
NATIVE_SAM3 = ("sam3", "sam3_video")
# GBC's own SAM 3 with no measured plan: 5090 bf16, 16 pairs per pass, exact graphs,
# 3.4 GiB allocated / 4.8 reserved
DEFAULT_SAM3_GB = 5.0
NATIVE_GEMMA = ("gemma3_text",)

# The plan's default estimates (weights and peak activations per device), by model
ESTIMATE_GB = {
    "google/owlv2-base-patch16-ensemble": 2.0,
    "google/owlv2-large-patch14-ensemble": 4.0,
    # bfloat16: 1.6 GiB of weights, 1.9 GiB at 4 crops of 16 texts in passes of 4 pairs
    "facebook/sam3": 3.5,
    "google/embeddinggemma-300m": 1.4,
    # bfloat16: 0.97 GiB of weights (MXFP8 ViT: 0.5), activations at 4 images per pass
    "pixai-labs/pixai-tagger-v1.0": 2.5,
}


def _engine(shared: dict, placements: list, dtype: str, settings: dict) -> TorchEngine:
    options = dict(settings.get("engine") or {})
    key = ("torch", tuple(placement.flat(placements)), dtype, tuple(sorted(options.items())))
    if key not in shared:
        shared[key] = TorchEngine(
            devices=placement.flat(placements),
            dtype=dtype,
            schedule=options.pop("schedule", "co_resident"),
            name=f"torch-{len([k for k in shared if k[0] == 'torch'])}",
            **options,
        )
    return shared[key]


def detect_costs(item) -> dict:
    """A detector request's costs from its inputs: one crop, one pair per text."""
    return {"crops": 1, "pairs": len(item.inputs["request"]["texts"])}


def embed_costs(item) -> dict:
    """An embedder request's costs: one text, its characters (a cheap stand-in for tokens)."""
    return {"texts": 1, "chars": len(item.inputs["text"])}


def tag_costs(item) -> dict:
    """A tagger request's costs: its picture and its crops."""
    return {"images": 1 + len(item.get("crops", {}))}


# Each lane's cost estimator and its revision (part of Togeari's execution identity: bump it
# when the estimate changes)
COSTS = {
    "detector": (detect_costs, "gbc-detect-crops-pairs/1"),
    "embedder": (embed_costs, "gbc-embed-texts-chars/1"),
    "tagger": (tag_costs, "gbc-tag-images/2"),
}


def former_of(name: str, model, settings: dict, budgets: dict):
    """The lane's batch former: ``size`` (requests) or ``budget`` (``budgets`` updated by the
    settings' ``budgets``)."""
    wait_ms = float(settings.get("wait_ms", 2.0))
    kind = settings.get("former", "size")
    if kind == "size":
        return SizeOrWait(model.max_batch, wait_ms)
    if kind != "budget":
        raise ValueError(f"former {kind!r}: size or budget")
    estimate, revision = COSTS[name]
    return BudgetFormer(
        budgets={**budgets, **(settings.get("budgets") or {})},
        estimate=estimate,
        size=model.max_batch,
        wait_ms=wait_ms,
        oversized="singleton",
        estimator_revision=revision,
    )


def _slot(
    name: str, model, settings: dict, shared: dict, default_dtype: str, arg, budgets: dict
) -> Slot:
    placements = placement.devices(settings.get("devices", "auto"))
    dtype = placement.dtype(settings.get("dtype", default_dtype), placements[0])
    engine = _engine(shared, placements, dtype, settings)
    engine.add_model(name, model, priority=int(settings.get("priority", 0)), devices=placements)
    gb = float(settings.get("memory_gb", ESTIMATE_GB.get(model_name(model), 2.0)))
    needs = [
        placement.Need(name, d, int(gb * placement.GIB), "weights and activations")
        for d in placement.flat(placements)
    ]
    return Slot(
        units=engine.units(name),
        former=former_of(name, model, settings, budgets),
        adapter=TorchAdapter(arg=arg),
        lane=dict(settings.get("lane") or {}),
        needs=needs,
    )


def model_name(model) -> str:
    return getattr(model, "model_name", "")


def native_sam3(name: str, settings: dict, shared: dict, common: dict) -> Slot:
    """GBC's own SAM 3, its caps and the plan's estimate from the card's memory plan
    (``detect.sam3.memory``) when one was measured: ``budget_gb`` fits ``max_pairs`` /
    ``max_crops`` within it; ``memory_gb`` defaults to the plan's estimate at the caps (+10%),
    else ``DEFAULT_SAM3_GB``."""
    placements = placement.devices(settings.get("devices", "auto"))
    first = placement.flat(placements)[0]
    dtype = placement.dtype(settings.get("dtype", "auto"), placements[0])
    box_source = settings.get("box_source", "pred")
    fp8 = fp8_recipe(first, settings.get("fp8"))
    text_device = settings.get("text_device")
    vision_graph = bool(settings.get("vision_graph", False))
    max_batch = int(settings.get("max_batch", 4))

    # A batch holds one crop per request: a vision pass never runs more than max_batch crops
    crops = min(int(settings.get("max_crops", 4)), max_batch)
    pairs = int(settings.get("max_pairs", 16))
    plan_name = memory.plan_name(dtype, fp8, text_device, box_source, vision_graph)
    plan = memory.load(first, plan_name)
    if plan is None and settings.get("budget_gb") is not None:
        raise ValueError(
            f"SAM 3 budget_gb: no memory plan {plan_name!r} for {first} "
            "(scripts/bench/detector/memory.py measures it)"
        )
    if plan is not None and settings.get("budget_gb") is not None:
        fitted = plan.fit(int(float(settings["budget_gb"]) * placement.GIB), crops, pairs)
        if fitted is None:
            raise ValueError(f"SAM 3 does not fit {settings['budget_gb']} GiB on {first}")
        crops, pairs = fitted
    model = NativeSam3Detector(
        name,
        max_batch=max_batch,
        box_source=box_source,
        mask_threshold=float(settings.get("mask_threshold", 0.5)),
        max_pairs=pairs,
        max_crops=crops,
        text_cache=int(settings.get("text_cache", 4096)),
        kernels=bool(settings.get("kernels", True)),
        fusion_graph=bool(settings.get("fusion_graph", True)),
        fusion_buckets=settings.get("fusion_buckets", "exact"),
        text_graph=bool(settings.get("text_graph", True)),
        fp8=fp8,
        fp8_encoder=bool(settings.get("fp8_encoder", False)),
        fp4_scales=settings.get("fp4_scales"),
        fp8_mlp=settings.get("fp8_mlp"),
        text_device=text_device,
        capture=settings.get("capture", "load"),
        retry_singular=bool(settings.get("retry_singular", True)),
        vision_graph=vision_graph,
        **common,
    )
    estimate = DEFAULT_SAM3_GB
    if plan is not None:
        estimate = plan.estimate(crops, pairs) * 1.1 / placement.GIB
    settings = {"memory_gb": estimate, **settings, "dtype": dtype}
    budgets = {"crops": crops, "pairs": pairs}
    return _slot("detector", model, settings, shared, dtype, None, budgets)


@entry("detect", "torch")
def detector(settings: dict, shared: dict) -> Slot:
    name = settings.get("model", "google/owlv2-base-patch16-ensemble")
    common = {
        "box_threshold": float(settings.get("box_threshold", 0.05)),
        "max_texts": int(settings.get("max_texts", 16)),
        "name": "detector",
    }
    if "sam3" in name.lower():
        impl = settings.get("impl", "auto")
        if impl == "auto":
            native = AutoConfig.from_pretrained(name).model_type in NATIVE_SAM3
            impl = "native" if native else "transformers"
        if impl == "native":
            return native_sam3(name, settings, shared, common)
        model = Sam3Detector(
            name,
            max_batch=int(settings.get("max_batch", 4)),
            max_units=int(settings.get("max_units", 4)),
            box_source=settings.get("box_source", "pred"),
            mask_threshold=float(settings.get("mask_threshold", 0.5)),
            retry_singular=bool(settings.get("retry_singular", True)),
            **common,
        )
    else:
        model = Owlv2Detector(name, max_batch=int(settings.get("max_batch", 8)), **common)
    budgets = {"crops": model.max_batch, "pairs": model.max_batch * model.max_texts}
    return _slot("detector", model, settings, shared, "float32", None, budgets)


@entry("embed", "torch")
def embedder(settings: dict, shared: dict) -> Slot:
    name = settings.get("model", "google/embeddinggemma-300m")
    impl = settings.get("impl", "auto")
    if impl == "auto":
        impl = "native" if AutoConfig.from_pretrained(name).model_type in NATIVE_GEMMA else "st"
    max_batch = int(settings.get("max_batch", 64))
    budgets = {"texts": max_batch, "chars": 4 * int(settings.get("graph_tokens", 4096))}
    if impl == "native":
        model = NativeEmbedder(
            name,
            max_batch=max_batch,
            name="embedder",
            kernels=bool(settings.get("kernels", True)),
            graphs=bool(settings.get("graphs", True)),
            residual=settings.get("residual", "float32"),
            capture=settings.get("capture", "load"),
            graph_tokens=int(settings.get("graph_tokens", 4096)),
        )
        # float32 keeps the PoC's vectors (5090: 1 - cos <= 1e-7, 1.9-8.3x the
        # sentence-transformers fp32 throughput); bfloat16 is 1.8-1.2x faster again, with a
        # 1 - cos tail to 2e-3 on short names
        return _slot("embedder", model, settings, shared, "float32", "text", budgets)
    model = SentenceEmbedder(name, max_batch=max_batch, name="embedder")
    return _slot("embedder", model, settings, shared, "float32", "text", budgets)


@entry("tag", "torch")
def tagger(settings: dict, shared: dict) -> Slot:
    name = settings.get("model", "pixai-labs/pixai-tagger-v1.0")
    placements = placement.devices(settings.get("devices", "auto"))
    max_batch = int(settings.get("max_batch", 4))
    model = NativeTagger(
        name,
        max_batch=max_batch,
        fp8=fp8_recipe(placement.flat(placements)[0], settings.get("fp8")),
        kernels=bool(settings.get("kernels", True)),
        vision_graph=bool(settings.get("vision_graph", True)),
    )
    return _slot("tagger", model, settings, shared, "auto", "request", {"images": max_batch})
