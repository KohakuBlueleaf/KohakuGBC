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
their parts and small objects that OWLv2 misses or boxes partly.

``SentenceEmbedder``: a sentence-transformers model (EmbeddingGemma by default), its pooled
embedding normalised, as ``SentenceTransformer.encode(..., normalize_embeddings=True)``.
Payload the text; result the vector (float32).
"""

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam3Model, Sam3Processor

from kohakugbc.engines.placement import LOADING


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
    ):
        self.model_name, self.box_threshold = model_name, box_threshold
        self.max_texts, self.max_batch, self.name = max_texts, max_batch, name
        self.max_units = max_units
        self.processor = self.model = None
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        self.processor = Sam3Processor.from_pretrained(self.model_name)
        with LOADING:
            model = Sam3Model.from_pretrained(self.model_name, dtype=dtype)
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
            "units": units,
            "pixels": pixels,
            "text": text,
            "sizes": [img.size for img in images],
            "texts": [len(p["request"]["texts"]) for p in payloads],
        }

    def forward(self, batch: dict) -> list[dict]:
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
            results += self.processor.post_process_instance_segmentation(
                outputs, threshold=self.box_threshold, mask_threshold=0.5, target_sizes=sizes
            )
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
