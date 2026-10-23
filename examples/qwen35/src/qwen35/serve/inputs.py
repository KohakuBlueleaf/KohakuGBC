"""A GBC chat request as the model's inputs, on the CPU (a pool of preparing threads runs this):
the chat template with thinking off, the images through the checkpoint's processor, the mrope
positions, and the prefix boundaries a snapshot can be taken at.

Boundaries: ``system`` ends where the user turn starts (every request with the same system
prompt shares it); ``images`` ends after the last image of the request (requests asking
different questions of the same pictures share it). A prefix key is a hash of the tokens and
of the images' bytes before the boundary.
"""

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

BOUNDARIES = ("system", "images")


@dataclass
class Prepared:
    ids: torch.Tensor  # [L] int64
    positions: torch.Tensor  # [3, L] int64
    next_position: int  # the mrope position of the first generated token
    pixels: torch.Tensor | None
    grid: torch.Tensor | None  # [images, 3]
    boundaries: list[tuple[int, bytes]] = field(default_factory=list)  # (length, key)
    schema: dict | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    image_end: int = 0  # one past the last image token (0 without images)


def chat_messages(request: dict) -> list[dict]:
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
    return messages


def pictures(request: dict) -> tuple[list[Image.Image], list[bytes]]:
    out, raw = [], []
    for _, picture in request["images"]:
        if picture is None:
            raise ValueError("an empty crop cannot be sent")
        with Image.open(io.BytesIO(picture["jpeg"])) as img:
            out.append(img.convert("RGB"))
        raw.append(picture["jpeg"])
    return out, raw


def prefix_key(ids: torch.Tensor, length: int, images: list[bytes]) -> bytes:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ids[:length].numpy()).tobytes())
    for jpeg in images:
        h.update(hashlib.sha256(jpeg).digest())
    return h.digest()


class Preparer:
    """Turns requests into ``Prepared``; ``boundaries`` names the snapshot points in use."""

    def __init__(self, processor, rope_index, shape, schema_of, boundaries=("system",)):
        unknown = set(boundaries) - set(BOUNDARIES)
        if unknown:
            raise ValueError(f"unknown snapshot boundaries {sorted(unknown)}; known {BOUNDARIES}")
        self.processor, self.rope_index, self.shape = processor, rope_index, shape
        self.schema_of = schema_of
        self.boundaries = tuple(boundaries)
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, request: dict) -> Prepared:
        text = self.processor.apply_chat_template(
            chat_messages(request),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        images, raw = pictures(request)
        enc = self.processor(
            text=[text], images=images or None, return_tensors="pt", return_mm_token_type_ids=True
        )
        ids = enc["input_ids"][0]
        grid = enc.get("image_grid_thw")
        types = enc.get("mm_token_type_ids")
        if types is None:
            types = (ids == self.shape.image_token).to(torch.int32)[None]
        positions, _ = self.rope_index(ids[None], types, image_grid_thw=grid)
        positions = positions[:, 0]
        prep = Prepared(
            ids=ids,
            positions=positions,
            next_position=int(positions.max()) + 1,
            pixels=enc.get("pixel_values"),
            grid=grid,
            schema=self.schema_of(request["schema"]) if request["schema"] else None,
            temperature=float(request["temperature"]),
            max_tokens=int(request["max_tokens"]),
        )
        image = (ids == self.shape.image_token).nonzero().flatten()
        prep.image_end = int(image[-1]) + 1 if image.numel() else 0
        prep.boundaries = self._boundaries(ids, raw)
        return prep

    def _boundaries(self, ids: torch.Tensor, raw: list[bytes]) -> list[tuple[int, bytes]]:
        out = []
        starts = (ids == self.im_start).nonzero().flatten().tolist()
        if "system" in self.boundaries and len(starts) > 1:
            out.append((starts[1], prefix_key(ids, starts[1], [])))
        if "images" in self.boundaries and raw:
            ends = (ids == self.shape.vision_end).nonzero().flatten().tolist()
            out.append((ends[-1] + 1, prefix_key(ids, ends[-1] + 1, raw)))
        return [b for b in out if 0 < b[0] < len(ids)]
