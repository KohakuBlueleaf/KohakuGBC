"""The vision models of GBC as Togeari torch models (``TorchModel``): batched forwards.

``Owlv2Detector``: open-vocabulary detection, as the PoC runs OWLv2 (``gbcpoc.detection.owlv2``):
every (crop, up to ``max_texts`` texts) of a batch is one row of one forward, rows padded to the
longest text list with blank queries whose boxes are dropped; boxes post-processed at
``box_threshold`` on the padded square, clamped to the crop. Payload ``{"patch", "request"}``;
result the detector's raw output ``{"boxes", "scores", "labels"}`` in crop pixels.

``Sam3Detector``: the same payloads and results through SAM 3 (``facebook/sam3``), promptable
concept detection: the vision encoder runs once per crop, then one decoder pass per (crop,
text), at most ``max_units`` pairs at once (each holds a copy of its crop's features); its
instances' boxes and scores above ``box_threshold``. On illustrated images it finds people,
their parts and small objects that OWLv2 misses or boxes partly. A SAM3-LiteText checkpoint
(``vil-uob/sam3-litetext-*``: SAM 3's vision encoder, a distilled text encoder) runs the same
way; the checkpoint's ``model_type`` picks the model class. ``box_source`` picks the boxes:

- ``pred`` (default): the model's predicted boxes, scaled to the crop, no mask work;
- ``mask``: the box of each kept instance's mask binarized at ``mask_threshold`` on the model's
  288^2 grid, scaled to the crop;
- ``mask_full``: the box of the mask bilinearly resized to the crop first (transformers'
  instance post-processing), resized a chunk of instances at a time (``mask_chunk_mib``).

An instance whose binarized mask is empty keeps its predicted box. Scores are the same in every
mode (the query score times the presence score).

``NativeSam3Detector``: the same payloads, results and box sources through GBC's own SAM 3
(``kohakugbc_models.detect.sam3``, ``sam3`` / ``sam3_video`` checkpoints).

``SentenceEmbedder``: a sentence-transformers model (EmbeddingGemma by default), its pooled
embedding normalised, as ``SentenceTransformer.encode(..., normalize_embeddings=True)``.
Payload the text; result the vector (float32).

``NativeEmbedder``: the same payloads and results through GBC's own EmbeddingGemma
(``kohakugbc_models.embed.gemma``): a batch's texts packed into one token stream, the pass replayed
from CUDA graphs.
"""

import inflect
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    Sam3LiteTextModel,
    Sam3Model,
    Sam3Processor,
)

from kohakugbc.engines.placement import LOADING
from kohakugbc_models.detect.boxes import BOX_SOURCES, mask_boxes
from kohakugbc_models.detect.sam3.model import Sam3Detector as OwnSam3
from kohakugbc_models.embed.gemma import EmbeddingGemma

# By the checkpoint's model_type (facebook/sam3 is the video config; its image model is Sam3Model)
SAM3_MODELS = {"sam3": Sam3Model, "sam3_video": Sam3Model, "sam3_lite_text": Sam3LiteTextModel}
INFLECT = inflect.engine()


def singular_phrase(text: str) -> str | None:
    """The phrase with its last word singular ("devil girls" -> "devil girl"); None when it is
    singular already."""
    words = text.split()
    one = INFLECT.singular_noun(words[-1]) if words else False
    return " ".join([*words[:-1], one]) if one and one != words[-1] else None


def retry_singular(raws: list[dict], payloads: list, answer) -> list[dict]:
    """Texts no box answered, asked again in their singular form (SAM 3 answers nothing for some
    plural phrases: "devil girls" none, "devil girl" 0.75); the boxes found join the text's own
    label. ``answer``: payloads to raw outputs."""
    again, where = [], []
    for i, (payload, raw) in enumerate(zip(payloads, raws)):
        found = set(raw["labels"])
        texts, labels = [], []
        for k, text in enumerate(payload["request"]["texts"]):
            one = singular_phrase(text)
            if k not in found and one is not None:
                texts.append(one)
                labels.append(k)
        if texts:
            again.append({**payload, "request": {**payload["request"], "texts": texts}})
            where.append((i, labels))
    if not again:
        return raws
    for (i, labels), raw in zip(where, answer(again)):
        raws[i]["boxes"] = list(raws[i]["boxes"]) + list(raw["boxes"])
        raws[i]["scores"] = list(raws[i]["scores"]) + list(raw["scores"])
        raws[i]["labels"] = list(raws[i]["labels"]) + [labels[k] for k in raw["labels"]]
    return raws


class Owlv2Detector:
    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        box_threshold: float = 0.05,
        max_texts: int = 16,
        max_batch: int = 8,
        name: str = "owlv2",
    ):
        self.model_name, self.box_threshold = model_name, box_threshold
        self.max_texts, self.max_batch, self.name = max_texts, max_batch, name
        self.processor = self.model = None
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        # The torchvision image processor: the slow one resizes differently, and its boxes
        # differ from the PoC's (which ran the fast one)
        self.processor = Owlv2Processor.from_pretrained(self.model_name, use_fast=True)
        with LOADING:
            model = Owlv2ForObjectDetection.from_pretrained(self.model_name, dtype=dtype)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype

    def to(self, device) -> None:
        self.model.to(device)
        self.device = device

    def example(self) -> list:
        return [{"patch": np.full((96, 128, 3), 127, np.uint8), "request": {"texts": ["object"]}}]

    def prepare(self, payloads: list) -> dict:
        images = [Image.fromarray(p["patch"]) for p in payloads]
        units = [
            (i, begin, p["request"]["texts"][begin : begin + self.max_texts])
            for i, p in enumerate(payloads)
            for begin in range(0, len(p["request"]["texts"]), self.max_texts)
        ]
        inputs = None
        if units:
            inputs = self.processor(
                text=[chunk for _, _, chunk in units],
                images=[images[i] for i, _, _ in units],
                return_tensors="pt",
            )
        return {
            "units": units,
            "inputs": inputs,
            "sizes": [img.size for img in images],
            "texts": [len(p["request"]["texts"]) for p in payloads],
        }

    def forward(self, batch: dict) -> list[dict]:
        per = [[([], []) for _ in range(n)] for n in batch["texts"]]
        units = batch["units"]
        if units:
            inputs = batch["inputs"].to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            outputs = self.model(**inputs)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    outputs[key] = value.float()

            # OWLv2 pads each crop to a square: boxes are relative to its longer side
            sides = [max(batch["sizes"][i]) for i, _, _ in units]
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.box_threshold, target_sizes=[(s, s) for s in sides]
            )
            for (i, begin, chunk), result in zip(units, results):
                width, height = batch["sizes"][i]
                for box, score, label in zip(
                    result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()
                ):
                    if label >= len(chunk):
                        continue
                    left, top, right, bottom = box
                    box = [max(0, left), max(0, top), min(width, right), min(height, bottom)]
                    per[i][begin + label][0].append(box)
                    per[i][begin + label][1].append(score)

        # Per crop: every text's boxes in text order, with the text's index as label
        raws = []
        for texts in per:
            raw = {"boxes": [], "scores": [], "labels": []}
            for label, (boxes, scores) in enumerate(texts):
                raw["boxes"] += boxes
                raw["scores"] += scores
                raw["labels"] += [label] * len(scores)
            raws.append(raw)
        return raws

    def split(self, out: list, n: int) -> list:
        return out

    def key(self, payload) -> None:
        return None


def _rows(output, index: torch.Tensor):
    """A model output with every tensor (and tuple of tensors) taken at ``index`` along its
    first dimension: one crop's vision features repeated for each of its texts."""
    out = {}
    for name, value in output.items():
        if torch.is_tensor(value):
            out[name] = value.index_select(0, index)
        elif isinstance(value, tuple) and value and torch.is_tensor(value[0]):
            out[name] = tuple(v.index_select(0, index) for v in value)
        else:
            out[name] = value
    return type(output)(**out)


class Sam3Detector:
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        box_threshold: float = 0.05,
        max_texts: int = 16,
        max_batch: int = 4,
        max_units: int = 4,
        name: str = "sam3",
        box_source: str = "pred",
        mask_threshold: float = 0.5,
        mask_chunk_mib: int = 256,
        retry_singular: bool = True,
    ):
        if box_source not in BOX_SOURCES:
            raise ValueError(f"box_source {box_source!r}: one of {BOX_SOURCES}")
        self.retry_singular = retry_singular
        self.model_name, self.box_threshold = model_name, box_threshold
        self.max_texts, self.max_batch, self.name = max_texts, max_batch, name
        self.max_units = max_units
        self.box_source, self.mask_threshold = box_source, mask_threshold
        self.mask_chunk_mib = mask_chunk_mib
        self.processor = self.model = None
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        self.processor = Sam3Processor.from_pretrained(self.model_name)
        model_class = SAM3_MODELS[AutoConfig.from_pretrained(self.model_name).model_type]
        with LOADING:
            model = model_class.from_pretrained(self.model_name, dtype=dtype)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype

    def to(self, device) -> None:
        self.model.to(device)
        self.device = device

    def example(self) -> list:
        return [{"patch": np.full((96, 128, 3), 127, np.uint8), "request": {"texts": ["object"]}}]

    def prepare(self, payloads: list) -> dict:
        images = [Image.fromarray(p["patch"]) for p in payloads]
        units = [
            (i, k, t) for i, p in enumerate(payloads) for k, t in enumerate(p["request"]["texts"])
        ]
        pixels = text = None
        if units:
            pixels = self.processor(images=images, return_tensors="pt")
            text = self.processor(text=[t for _, _, t in units], return_tensors="pt")
        return {
            "payloads": payloads,
            "units": units,
            "pixels": pixels,
            "text": text,
            "sizes": [img.size for img in images],
            "texts": [len(p["request"]["texts"]) for p in payloads],
        }

    def forward(self, batch: dict) -> list[dict]:
        raws = self.answer(batch)
        if self.retry_singular:
            raws = retry_singular(raws, batch["payloads"], lambda p: self.answer(self.prepare(p)))
        return raws

    def answer(self, batch: dict) -> list[dict]:
        raws = [{"boxes": [], "scores": [], "labels": []} for _ in batch["texts"]]
        units = batch["units"]
        if not units:
            return raws
        pixel_values = batch["pixels"]["pixel_values"].to(self.device, self.dtype)
        vision = self.model.get_vision_features(pixel_values=pixel_values)
        text = batch["text"].to(self.device)
        results = []
        # Each (crop, text) pass holds a copy of its crop's features (about 0.35 GiB at bfloat16):
        # at most max_units of them at once
        for start in range(0, len(units), self.max_units):
            part = units[start : start + self.max_units]
            crop = torch.tensor([i for i, _, _ in part], device=self.device)
            outputs = self.model(
                vision_embeds=_rows(vision, crop),
                input_ids=text["input_ids"][start : start + len(part)],
                attention_mask=text["attention_mask"][start : start + len(part)],
            )
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    outputs[key] = value.float()
            sizes = [[batch["sizes"][i][1], batch["sizes"][i][0]] for i, _, _ in part]
            results += self.boxes(outputs, sizes)
            del outputs
        for (i, label, _), result in zip(units, results):
            width, height = batch["sizes"][i]
            for box, score in zip(result["boxes"].tolist(), result["scores"].tolist()):
                left, top, right, bottom = box
                raws[i]["boxes"].append(
                    [max(0, left), max(0, top), min(width, right), min(height, bottom)]
                )
                raws[i]["scores"].append(score)
                raws[i]["labels"].append(label)
        return raws

    def boxes(self, outputs, sizes: list) -> list[dict]:
        """Per pass row: the kept instances' ``scores`` and ``boxes`` (crop pixels)."""
        found = self.processor.post_process_object_detection(
            outputs, threshold=self.box_threshold, target_sizes=sizes
        )
        if self.box_source == "pred":
            return found
        # The kept set as post_process_object_detection computes it, for the masks to match
        scores = outputs.pred_logits.sigmoid()
        if outputs.presence_logits is not None:
            scores = scores * outputs.presence_logits.sigmoid()
        full = self.box_source == "mask_full"
        for row, (result, size) in enumerate(zip(found, sizes)):
            keep = scores[row] > self.box_threshold
            chunk = max(1, (self.mask_chunk_mib << 20) // (size[0] * size[1] * 5))
            boxes, empty = mask_boxes(
                outputs.pred_masks[row][keep], size, self.mask_threshold, full, chunk
            )
            result["boxes"] = torch.where(empty[:, None], result["boxes"], boxes)
        return found

    def split(self, out: list, n: int) -> list:
        return out

    def key(self, payload) -> None:
        return None


class NativeSam3Detector:
    """GBC's own SAM 3 (``kohakugbc_models.detect.sam3``) behind the same payloads and results as
    ``Sam3Detector``: the crops' pixels made on the host in ``prepare``; each crop's vision
    features once, each unique text once (cached), (crop, text) pairs ``max_pairs`` per fusion
    pass reading their crop's features in place. ``kernels``: the kernel registry's ops;
    ``fusion_graph`` / ``fusion_buckets`` / ``vision_graph``: the fusion and vision passes
    replayed from CUDA graphs."""

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        box_threshold: float = 0.05,
        max_texts: int = 16,
        max_batch: int = 4,
        name: str = "sam3",
        box_source: str = "pred",
        mask_threshold: float = 0.5,
        max_pairs: int = 16,
        max_crops: int = 4,
        text_cache: int = 4096,
        kernels: bool = True,
        fusion_graph: bool = True,
        fusion_buckets: str = "exact",
        text_graph: bool = True,
        fp8: str | None = None,
        fp8_encoder: bool = False,
        text_device: str | None = None,
        capture: str = "load",
        retry_singular: bool = True,
        vision_graph: bool = False,
        fp4_scales: str | None = None,
        fp8_mlp: str | None = None,
    ):
        self.model_name, self.box_threshold = model_name, box_threshold
        self.max_texts, self.max_batch, self.name = max_texts, max_batch, name
        self.mask_threshold, self.retry_singular = mask_threshold, retry_singular
        self.detector = OwnSam3(
            model_name,
            box_source=box_source,
            max_pairs=max_pairs,
            max_crops=max_crops,
            text_cache=text_cache,
            kernels=kernels,
            fusion_graph=fusion_graph,
            fusion_buckets=fusion_buckets,
            text_graph=text_graph,
            fp8=fp8,
            fp8_encoder=fp8_encoder,
            text_device=text_device,
            capture=capture,
            max_texts=max_texts * max_batch,
            vision_graph=vision_graph,
            fp4_scales=fp4_scales,
            fp8_mlp=fp8_mlp,
        )
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        with LOADING:
            self.detector.load(device, dtype)
        self.device, self.dtype = device, dtype

    def to(self, device) -> None:
        self.detector.to(device)
        self.device = device

    def example(self) -> list:
        return [{"patch": np.full((96, 128, 3), 127, np.uint8), "request": {"texts": ["object"]}}]

    def prepare(self, payloads: list) -> dict:
        pixels = None
        if any(p["request"]["texts"] for p in payloads):
            pixels = self.detector.pin(self.detector.preprocess([p["patch"] for p in payloads]))
        return {"payloads": payloads, "pixels": pixels}

    def forward(self, batch: dict) -> list[dict]:
        raws = self.detector.run(
            batch["payloads"], self.box_threshold, self.mask_threshold, pixels=batch["pixels"]
        )
        if self.retry_singular:
            raws = retry_singular(
                raws,
                batch["payloads"],
                lambda p: self.detector.run(p, self.box_threshold, self.mask_threshold),
            )
        return raws

    def split(self, out: list, n: int) -> list:
        return out

    def key(self, payload) -> None:
        return None


class SentenceEmbedder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        max_batch: int = 64,
        name: str = "embedder",
    ):
        self.model_name, self.max_batch, self.name = model_name, max_batch, name
        self.model: SentenceTransformer | None = None
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        with LOADING:
            self.model = SentenceTransformer(self.model_name, device=device)
        self.model.to(dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype

    def to(self, device) -> None:
        self.model.to(device)
        self.device = device

    def example(self) -> list:
        return ["red dress"]

    def prepare(self, payloads: list) -> dict:
        return self.model.preprocess(list(payloads))

    def forward(self, batch: dict) -> np.ndarray:
        features = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = self.model(features)["sentence_embedding"]
        return torch.nn.functional.normalize(out.float(), p=2, dim=1).cpu().numpy()

    def split(self, out: np.ndarray, n: int) -> list:
        return [out[i] for i in range(n)]

    def key(self, payload) -> None:
        return None


class NativeEmbedder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        max_batch: int = 64,
        name: str = "embedder",
        kernels: bool = True,
        graphs: bool = True,
        residual: str = "float32",
        capture: str = "load",
        graph_tokens: int = 4096,
    ):
        self.model_name, self.max_batch, self.name = model_name, max_batch, name
        self.model = EmbeddingGemma(
            model_name,
            max_batch=max_batch,
            kernels=kernels,
            graphs=graphs,
            residual=residual,
            capture=capture,
            graph_tokens=graph_tokens,
        )
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        with LOADING:
            self.model.load(device, dtype)
        self.device, self.dtype = device, dtype

    def to(self, device) -> None:
        self.model.to(device)
        self.device = device

    def example(self) -> list:
        return ["red dress"]

    def prepare(self, payloads: list) -> tuple:
        return self.model.pack(self.model.tokenize(list(payloads)))

    def forward(self, batch: tuple) -> np.ndarray:
        return self.model.run(batch).cpu().numpy()

    def split(self, out: np.ndarray, n: int) -> list:
        return [out[i] for i in range(n)]

    def key(self, payload) -> None:
        return None
