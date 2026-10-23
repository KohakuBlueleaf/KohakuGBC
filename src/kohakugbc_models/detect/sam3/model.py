"""SAM 3 image detection, run as GBC runs it: crops and text prompts in, the kept instances'
boxes and scores out.

Loads transformers' ``Sam3Model`` weights (``detector_model.*`` of a ``sam3_video`` checkpoint,
or a plain ``sam3`` one) into GBC's own layers (``vision``, ``fusion``, ``masks``) and
transformers' CLIP text encoder. What it does once that transformers' forward does per pair:

- the vision encoder once per crop, the neck at the levels the box source reads (level 2 for
  ``pred``; levels 0-2 for the mask sources);
- the text encoder once per unique text, its features cached (``text_cache`` texts, least
  recently used dropped first);
- the mask decoder only for the mask box sources; no semantic head;
- pairs gather their crop's level-2 features (``[256, 72, 72]`` each); the neck's position
  encoding is shared by every crop.

``detect`` runs (crop, text) pairs ``max_pairs`` at a time and returns per pair the query
logits, the presence logit, the boxes (xyxy, 0..1) and, for the mask sources, the mask logits.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, CLIPTextModelWithProjection, Sam3Config, Sam3Processor

from kohakugbc_models.detect.boxes import mask_boxes
from kohakugbc_models.detect.sam3.fusion import Decoder, Encoder, Scoring
from kohakugbc_models.detect.sam3.masks import MaskDecoder
from kohakugbc_models.detect.sam3.ops import EAGER, Ops
from kohakugbc_models.detect.sam3.vision import VisionEncoder, linear
from kohakugbc_models.kernels import fp8
from kohakugbc_models.kernels.graphs import Graphs

PREFIX = "detector_model."
OWNED = (
    "vision_encoder.",
    "text_encoder.",
    "text_projection.",
    "detr_encoder.",
    "detr_decoder.",
    "dot_product_scoring.",
    "mask_decoder.",
)
FILES = ["config.json", "*.safetensors", "*.json", "*.txt"]


def detector_config(path: Path) -> Sam3Config:
    config = AutoConfig.from_pretrained(path)
    inner = getattr(config, "detector_config", None)
    if inner is None:
        return config
    return Sam3Config(**inner) if isinstance(inner, dict) else inner


class Sam3Image(nn.Module):
    def __init__(self, config: Sam3Config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config.vision_config)
        self.text_encoder = CLIPTextModelWithProjection(config.text_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, config.detr_encoder_config.hidden_size
        )
        self.detr_encoder = Encoder(config.detr_encoder_config)
        self.detr_decoder = Decoder(config.detr_decoder_config)
        dec = config.detr_decoder_config
        self.dot_product_scoring = Scoring(dec.hidden_size, dec.intermediate_size)
        self.mask_decoder = MaskDecoder(config.mask_decoder_config)

    def bind_ops(self, bound: Ops) -> None:
        """Every attention layer calls ``bound`` (the eager ops, or the registry's)."""
        for module in self.modules():
            if hasattr(module, "ops"):
                module.ops = bound

    def fuse(self, vision, pos, text, valid) -> tuple:
        """One fusion pass over pairs: level-2 vision ``[P, C, h, w]``, its positions, text
        ``[P, S, C]`` and its valid tokens ``[P, S]``; the fused vision, the last decoder
        layer's normalized queries, boxes, presence logits and query logits."""
        mask = valid[:, None, None, :]
        fused = self.detr_encoder(vision, pos, text, mask)
        height, width = vision.shape[-2:]
        queries, boxes, presence = self.detr_decoder(
            fused, pos.flatten(2).transpose(1, 2), text, mask, height, width
        )
        return fused, queries, boxes, presence, self.dot_product_scoring(queries, text, valid)

    @torch.no_grad()
    def load_weights(self, files: list) -> None:
        """Every owned tensor from the checkpoint files; a missing one is an error."""
        own = self.state_dict()
        found = set()
        for file in files:
            with safe_open(str(file), "pt") as f:
                for key in f.keys():  # noqa: SIM118 (a safetensors handle is not iterable)
                    name = key[len(PREFIX) :] if key.startswith(PREFIX) else key
                    if name in own and name.startswith(OWNED):
                        own[name].copy_(f.get_tensor(key))
                        found.add(name)
        missing = sorted(set(own) - found)
        if missing:
            raise ValueError(f"checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")


class FusionGraphs(Graphs):
    """``fuse`` replayed from one CUDA graph per pass size: the boxes, presence and query
    logits come back cloned (rows outlive the next replay), the fused vision and queries as
    the graph's own tensors (read before the next pass)."""

    def __init__(self, fuse):
        super().__init__(fuse, clone=(2, 3, 4))

    def __call__(self, *inputs) -> tuple:
        return super().__call__(inputs[0].shape[0], *inputs)

    def warm(self, *inputs) -> None:
        super().warm(inputs[0].shape[0], *inputs)


class Sam3Detector:
    """``Sam3Image`` with its processor and text cache: what the detect engine calls.

    The vision encoder runs eager or ``torch.compile``d (``vision_compile``: a compile mode).
    The fusion pass runs eager, ``torch.compile``d (``fusion_compile``) or replayed from CUDA
    graphs (``fusion_graph``, CUDA only); either of the last two runs each pass at its bucket
    (``fusion_buckets``: ``pow2``, padded to the next power of two, at most log2(max_pairs) + 1
    graphs; ``exact``, the pair count itself, up to ``max_pairs`` graphs and no padded work).
    ``text_graph``: the text encoder replayed from CUDA graphs per power-of-two count of new
    texts. ``fp8``: the ViT's linears in fp8 (``row`` / ``mx``, ``kernels.fp8``), their inputs
    quantized by the producing norm / GELU / attention; ``fp8_encoder``: the DETR encoder's MLPs
    too (input norm and ReLU emitting fp8). ``text_device``: where the text encoder
    runs (``cpu``: fp32 on the host, its device memory freed; default the model's device).
    ``capture``: ``load`` captures every fusion bucket and every text count up to ``max_texts``
    new texts when placed on a CUDA device, then runs other shapes uncaptured; ``lazy`` captures
    each on its first call.
    """

    def __init__(
        self,
        model: str = "facebook/sam3",
        box_source: str = "pred",
        max_pairs: int = 16,
        max_crops: int = 4,
        text_cache: int = 4096,
        fusion_compile: str | None = None,
        kernels: bool = False,
        fusion_graph: bool = False,
        fusion_buckets: str = "pow2",
        vision_compile: str | None = None,
        text_graph: bool = False,
        fp8: str | None = None,
        text_device: str | None = None,
        fp8_encoder: bool = False,
        capture: str = "load",
        max_texts: int = 64,
    ):
        if capture not in ("load", "lazy"):
            raise ValueError(f"capture {capture!r}: load or lazy")
        if fusion_compile and fusion_graph:
            raise ValueError("fusion_compile and fusion_graph are exclusive")
        if fusion_buckets not in ("pow2", "exact"):
            raise ValueError(f"fusion_buckets {fusion_buckets!r}: pow2 or exact")
        self.fusion_buckets, self.vision_compile = fusion_buckets, vision_compile
        self.text_graph, self.text_graphs = text_graph, None
        self.fp8, self.text_device, self.fp8_encoder = fp8, text_device, fp8_encoder
        self.capture, self.max_texts = capture, max_texts
        self.model_name, self.box_source = model, box_source
        self.max_pairs, self.max_crops, self.text_cache = max_pairs, max_crops, text_cache
        self.fusion_compile, self.kernels, self.fusion_graph = fusion_compile, kernels, fusion_graph
        self.model = self.processor = self.positions = self.fuse = self.vision = None
        self.texts: OrderedDict[str, tuple] = OrderedDict()
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        path = Path(self.model_name)
        if not path.is_dir():
            path = Path(snapshot_download(self.model_name, allow_patterns=FILES))
        model = Sam3Image(detector_config(path))
        model.load_weights(sorted(path.glob("*.safetensors")))
        self.model = model.to(dtype=dtype).eval().requires_grad_(False)
        self.processor = Sam3Processor.from_pretrained(path)
        self.dtype = dtype
        self.to(device)

    def to(self, device) -> None:
        """Place the model on ``device``: the tables, the fused q / k / v weights, the bound ops
        and the compiled or graphed passes are rebuilt there; the text cache starts empty."""
        self.model.to(device)
        if self.text_device:
            # The text encoder on its own device, in fp32 on the host: the cache serves repeats
            text_dtype = torch.float32 if self.text_device == "cpu" else self.dtype
            self.model.text_encoder.to(self.text_device, text_dtype)
            self.model.text_projection.to(self.text_device, text_dtype)
        vision = self.model.vision_encoder
        vision.backbone.rope(device)
        vision.backbone.fuse_qkv()
        vision.neck.positions(vision.backbone.grid, device, self.dtype)
        # fp8 weights quantize once on a device that runs them; once quantized they move to any
        # device, which then keeps them without running them (``parked``)
        self.parked = None
        if self.fp8:
            why = fp8.supports(device, self.fp8)
            if why is not None:
                if vision.backbone.layers[0].recipe is None:
                    raise ValueError(f"fp8 {self.fp8!r} on {device}: {why}")
                self.parked = f"fp8 {self.fp8!r} weights parked on {device}: {why}"
            vision.backbone.quantize(self.fp8, device)
            if self.fp8_encoder:
                self.model.detr_encoder.quantize(self.fp8, device)
        self.model.detr_decoder.frequencies(self.dtype, device)
        self.model.bind_ops(Ops.resolved(device, self.dtype) if self.kernels else EAGER)
        self.device = device
        self.texts.clear()
        self.vision = self.model.vision_encoder
        if self.vision_compile:
            self.vision = torch.compile(self.vision, mode=self.vision_compile, dynamic=False)
        self.fuse = self.model.fuse
        if self.fusion_compile:
            self.fuse = torch.compile(self.model.fuse, mode=self.fusion_compile, dynamic=False)
        if self.fusion_graph and torch.device(device).type == "cuda":
            self.fuse = FusionGraphs(self.model.fuse)
        self.text_graphs = None
        if self.text_graph and torch.device(self.text_device or device).type == "cuda":
            self.text_graphs = Graphs(self.text_features, clone=(0,))
        if self.capture == "load" and not self.parked and torch.device(device).type == "cuda":
            self.capture_all()

    def capture_all(self) -> None:
        """Every crop count's vision pass run once (its ops tuned), every fusion bucket and text
        count captured, then both graph sets frozen."""
        with torch.inference_mode():
            pixels = self.preprocess([np.zeros((64, 64, 3), np.uint8)] * self.max_crops)
            for n in range(1, self.max_crops + 1):
                crops = self.encode_pixels(pixels[:n])
            tokens = self.processor(text=["object"], return_tensors="pt").to(
                self.text_device or self.device
            )
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            if self.text_graphs is not None:
                n = 1
                while n <= self.max_texts:
                    many = (ids.expand(n, -1), mask.expand(n, -1))
                    self.text_graphs.warm(tuple(many[0].shape), *many)
                    n *= 2
                self.text_graphs.freeze()
            if isinstance(self.fuse, FusionGraphs):
                text = self.text_features(ids, mask).to(self.device, self.dtype)
                valid = mask.to(self.device).bool()
                sizes = {self.bucket(p) for p in range(1, self.max_pairs + 1)}
                for n in sorted(sizes):
                    vision = crops[0][2][None].expand(n, -1, -1, -1).contiguous()
                    pos = self.positions.expand(n, -1, -1, -1)
                    self.fuse.warm(vision, pos, text.expand(n, -1, -1), valid.expand(n, -1))
                self.fuse.freeze()
        self.texts.clear()

    def bucket(self, pairs: int) -> int:
        """The pass size a compiled or graphed fusion runs ``pairs`` at (static shapes, one
        graph per bucket); the pair count itself when issued per op or ``exact``."""
        if not (self.fusion_compile or self.fusion_graph) or self.fusion_buckets == "exact":
            return pairs
        return min(self.max_pairs, 1 << (pairs - 1).bit_length())

    @property
    def levels(self) -> tuple:
        return (2,) if self.box_source == "pred" else (0, 1, 2)

    def preprocess(self, patches: list) -> torch.Tensor:
        """The crops as the vision encoder's pixels ``[N, 3, S, S]`` (fp32, on the host)."""
        images = [Image.fromarray(p) for p in patches]
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def encode_pixels(self, pixels: torch.Tensor) -> list[dict]:
        """Per crop ``{level: features [C, h, w]}``, ``max_crops`` crops per vision pass; the
        level-2 positions kept for the fusion passes."""
        if self.parked:
            raise RuntimeError(self.parked)
        out = []
        for start in range(0, len(pixels), self.max_crops):
            part = pixels[start : start + self.max_crops].to(self.device, self.dtype)
            levels = self.vision(part, self.levels)
            out += [{lvl: feats[i] for lvl, (feats, _) in levels.items()} for i in range(len(part))]
            self.positions = levels[2][1][:1]
        return out

    def encode_images(self, patches: list) -> list[dict]:
        return self.encode_pixels(self.preprocess(patches))

    def text_features(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The text encoder and its projection: ``[N, S, C]``."""
        hidden = self.model.text_encoder(
            input_ids=ids, attention_mask=mask, return_dict=True
        ).last_hidden_state
        return linear(self.model.text_projection, hidden)

    def encode_texts(self, texts: list[str]) -> None:
        """Encode the texts not cached yet, in one batch, into ``self.texts``."""
        new = [t for t in dict.fromkeys(texts) if t not in self.texts]
        if new:
            tokens = self.processor(text=new, return_tensors="pt").to(
                self.text_device or self.device
            )
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            if self.text_graphs is not None:
                # One graph per power-of-two count: the last text repeated, its rows cut
                pad = (1 << (len(new) - 1).bit_length()) - len(new)
                ids = torch.cat([ids, ids[-1:].expand(pad, -1)])
                mask = torch.cat([mask, mask[-1:].expand(pad, -1)])
                feats = self.text_graphs(tuple(ids.shape), ids, mask)
            else:
                feats = self.text_features(ids, mask)
            feats = feats.to(self.device, self.dtype)
            mask = mask.to(self.device)
            for i, text in enumerate(new):
                self.texts[text] = (feats[i], mask[i].bool())
        for text in texts:
            self.texts.move_to_end(text)
        while len(self.texts) > self.text_cache:
            self.texts.popitem(last=False)

    def detect(self, crops: list[dict], pairs: list[tuple], mask_threshold: float = 0.5) -> list:
        """``pairs`` of (crop index, text): per pair ``logits [Q]``, ``presence [1]``,
        ``boxes [Q, 4]`` (xyxy, 0..1); for ``mask`` each query's mask box on the mask grid
        ``mask_box [Q, 4]`` (int32, at ``mask_threshold``), ``mask_empty [Q]`` and the grid size
        ``mask_grid``, the logits never written; for ``mask_full`` the logits ``masks [Q, H0,
        W0]``."""
        self.encode_texts([t for _, t in pairs])
        m = self.model
        out = []
        for start in range(0, len(pairs), self.max_pairs):
            part = pairs[start : start + self.max_pairs]
            # A compiled pass runs at its bucket size: the last pair repeated, outputs cut
            padded = part + [part[-1]] * (self.bucket(len(part)) - len(part))
            vision = torch.stack([crops[i][2] for i, _ in padded])
            pos = self.positions.expand(len(padded), -1, -1, -1)
            text = torch.stack([self.texts[t][0] for _, t in padded])
            valid = torch.stack([self.texts[t][1] for _, t in padded])
            fused, queries, boxes, presence, logits = self.fuse(vision, pos, text, valid)
            extra = {}
            if self.box_source != "pred":
                levels = [torch.stack([crops[i][lvl] for i, _ in padded]) for lvl in (0, 1, 2)]
                args = (queries, levels, fused, text, valid[:, None, None, :])
                if self.box_source == "mask":
                    grid, empty = m.mask_decoder.boxes(*args, mask_threshold)
                    size = levels[0].shape[-2:]
                    extra = {"mask_box": grid, "mask_empty": empty, "mask_grid": [size] * len(grid)}
                else:
                    extra = {"masks": m.mask_decoder(*args)}
            for j in range(len(part)):
                row = {"logits": logits[j], "presence": presence[j], "boxes": boxes[j]}
                row |= {k: v[j] for k, v in extra.items()}
                out.append(row)
        return out

    def run(
        self,
        payloads: list,
        threshold: float = 0.05,
        mask_threshold: float = 0.5,
        pixels: torch.Tensor | None = None,
    ) -> list:
        """The detect engine's call: payloads ``{"patch", "request": {"texts"}}`` to raw
        detections ``{"boxes", "scores", "labels"}`` per crop (crop pixels; label = the text's
        index), the instances whose score (query x presence, after the sigmoids) passes
        ``threshold``. ``pixels``: the payloads' ``preprocess``, when already made."""
        raws = [{"boxes": [], "scores": [], "labels": []} for _ in payloads]
        pairs = [(i, t) for i, p in enumerate(payloads) for t in p["request"]["texts"]]
        if not pairs:
            return raws
        labels = [k for p in payloads for k in range(len(p["request"]["texts"]))]
        if pixels is None:
            pixels = self.preprocess([p["patch"] for p in payloads])
        rows = self.detect(self.encode_pixels(pixels), pairs, mask_threshold)
        scores = torch.stack([r["logits"] for r in rows]).float().sigmoid()
        scores = scores * torch.stack([r["presence"] for r in rows]).float().sigmoid()
        sizes = torch.tensor(
            [[p["patch"].shape[1], p["patch"].shape[0]] * 2 for p in payloads], dtype=torch.float32
        )[[i for i, _ in pairs]].to(scores.device)
        boxes = torch.stack([r["boxes"] for r in rows]).float() * sizes[:, None, :]
        keep = scores > threshold
        if self.box_source == "mask":
            for j, row in enumerate(rows):
                height, width = payloads[pairs[j][0]]["patch"].shape[:2]
                grid_h, grid_w = row["mask_grid"]
                scale = torch.tensor([width / grid_w, height / grid_h] * 2, device=scores.device)
                found = row["mask_box"].float() * scale
                boxes[j] = torch.where(row["mask_empty"][:, None], boxes[j], found)
        elif self.box_source == "mask_full":
            for j, row in enumerate(rows):
                height, width = payloads[pairs[j][0]]["patch"].shape[:2]
                chunk = max(1, (256 << 20) // (height * width * 5))
                found, empty = mask_boxes(
                    row["masks"][keep[j]], (height, width), mask_threshold, True, chunk
                )
                boxes[j][keep[j]] = torch.where(empty[:, None], boxes[j][keep[j]], found)
        scores, boxes, keep = scores.cpu(), boxes.cpu(), keep.cpu()
        for j, ((i, _), label) in enumerate(zip(pairs, labels)):
            height, width = payloads[i]["patch"].shape[:2]
            for box, score in zip(boxes[j][keep[j]].tolist(), scores[j][keep[j]].tolist()):
                left, top, right, bottom = box
                raws[i]["boxes"].append(
                    [max(0, left), max(0, top), min(width, right), min(height, bottom)]
                )
                raws[i]["scores"].append(score)
                raws[i]["labels"].append(label)
        return raws
