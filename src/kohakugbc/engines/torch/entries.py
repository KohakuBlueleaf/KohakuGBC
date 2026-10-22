"""The ``torch`` entries of the ``detect`` and ``embed`` slots: GBC's vision models on Togeari's
torch engine.

Settings: ``model`` (a checkpoint), ``devices`` (``"auto"``, a device, or a list: one unit per
device, a list inside the list one model split across those devices), ``dtype`` (float32 by
default: OWLv2 in float16 gives batched scores up to 0.085 from single ones, float32 within
1.3e-4; float16 with ``max_batch`` 1 is the PoC's own run), ``max_batch``, ``wait_ms`` (the batch
former's wait), ``memory_gb`` (the plan's estimate per device), ``engine``
(``schedule``, ``isolation``, ``slots``, ``memory_cap_gb``, ``prep_workers``; slots whose models
share devices, dtype and these settings share one engine). Detector: ``box_threshold`` (0.05),
``max_texts`` (16).
"""

from togeari.engines.torch import TorchAdapter, TorchEngine
from togeari.policies import SizeOrWait

from kohakugbc.engines import placement
from kohakugbc.engines.registry import Slot, entry
from kohakugbc.engines.torch.modules import Owlv2Detector, SentenceEmbedder

# The plan's default estimates (weights and peak activations per device), by model
ESTIMATE_GB = {
    "google/owlv2-base-patch16-ensemble": 2.0,
    "google/owlv2-large-patch14-ensemble": 4.0,
    "google/embeddinggemma-300m": 1.4,
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


def _slot(name: str, model, settings: dict, shared: dict, default_dtype: str, arg) -> Slot:
    placements = placement.devices(settings.get("devices", "auto"))
    dtype = placement.dtype(settings.get("dtype", default_dtype), placements[0])
    engine = _engine(shared, placements, dtype, settings)
    engine.add_model(name, model, devices=placements)
    gb = float(settings.get("memory_gb", ESTIMATE_GB.get(model_name(model), 2.0)))
    needs = [
        placement.Need(name, d, int(gb * placement.GIB), "weights and activations")
        for d in placement.flat(placements)
    ]
    return Slot(
        units=engine.units(name),
        former=SizeOrWait(model.max_batch, float(settings.get("wait_ms", 2.0))),
        adapter=TorchAdapter(arg=arg),
        lane=dict(settings.get("lane") or {}),
        needs=needs,
    )


def model_name(model) -> str:
    return getattr(model, "model_name", "")


@entry("detect", "torch")
def detector(settings: dict, shared: dict) -> Slot:
    model = Owlv2Detector(
        settings.get("model", "google/owlv2-base-patch16-ensemble"),
        box_threshold=float(settings.get("box_threshold", 0.05)),
        max_texts=int(settings.get("max_texts", 16)),
        max_batch=int(settings.get("max_batch", 8)),
        name="detector",
    )
    return _slot("detector", model, settings, shared, "float32", None)


@entry("embed", "torch")
def embedder(settings: dict, shared: dict) -> Slot:
    model = SentenceEmbedder(
        settings.get("model", "google/embeddinggemma-300m"),
        max_batch=int(settings.get("max_batch", 64)),
        name="embedder",
    )
    return _slot("embedder", model, settings, shared, "float32", "text")
