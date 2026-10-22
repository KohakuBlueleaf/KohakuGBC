"""The vision models of GBC as Togeari torch models (``TorchModel``): batched forwards.

``Owlv2Detector``: open-vocabulary detection, as the PoC runs OWLv2 (``gbcpoc.detection.owlv2``):
every (crop, up to ``max_texts`` texts) of a batch is one row of one forward, rows padded to the
longest text list with blank queries whose boxes are dropped; boxes post-processed at
``box_threshold`` on the padded square, clamped to the crop. Payload ``{"patch", "request"}``;
result the detector's raw output ``{"boxes", "scores", "labels"}`` in crop pixels.

``SentenceEmbedder``: a sentence-transformers model (EmbeddingGemma by default), its pooled
embedding normalised, as ``SentenceTransformer.encode(..., normalize_embeddings=True)``.
Payload the text; result the vector (float32).
"""

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Owlv2ForObjectDetection, Owlv2Processor


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
