"""Detector requests and the selection of their boxes.

A request is ``{"key", "texts", "box"}``: open-vocabulary texts to find in the crop of ``box``
(None: the whole image). The detector answers raw boxes in crop pixels: ``{"boxes": [[l, t, r,
b]], "scores": [s], "labels": [text index]}``. Selection is ml-gbc's ``DetectionTextInput
.select_bboxes`` per text, computed in float32 like torch and torchvision's CPU NMS, so the
boxes it keeps are bit for bit those ml-gbc keeps.
"""

import numpy as np

from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes.specs import trace

SCORE_THRESHOLD = 0.1
NMS_SINGLE, NMS_MULTIPLE = 0.05, 0.2
HIGH_CONF_TOLERANCE = 0.05
MIN_ABS_AREA = 5000
MAX_REL_AREA_ENTITY = 0.8


def request(image_key: str, box, texts: list[str]) -> dict:
    return {"key": trace.detect_key(image_key, box, texts), "texts": list(texts), "box": box}


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """torchvision's CPU ``nms``: indices kept, highest score first, IoU in float32."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for n, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order[n + 1 :]:
            if suppressed[j]:
                continue
            w = max(np.float32(0), min(x2[i], x2[j]) - max(x1[i], x1[j]))
            h = max(np.float32(0), min(y2[i], y2[j]) - max(y1[i], y1[j]))
            inter = w * h
            if float(inter / (areas[i] + areas[j] - inter)) > threshold:
                suppressed[j] = True
    return keep


def select(
    raw_boxes: list,
    raw_scores: list,
    width: int,
    height: int,
    single: bool,
    max_rel_area: float | None,
    topk: int,
    score: float = SCORE_THRESHOLD,
    nms_iou: float = NMS_MULTIPLE,
    nms_iou_single: float = NMS_SINGLE,
    drop_containers: bool = True,
) -> list[list]:
    """The boxes one text keeps, relative to the crop, as ``[l, t, r, b, confidence]``: scores
    above ``score``, NMS at ``nms_iou`` (``nms_iou_single`` for a single object), a box that
    contains another dropped (``drop_containers``); the PoC's selection by default."""
    boxes = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(raw_scores, dtype=np.float32)

    # Score, then area thresholds (float32 against float32 thresholds, as torch compares)
    keep = scores > np.float32(score)
    boxes, scores = boxes[keep], scores[keep]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area > np.float32(MIN_ABS_AREA)
    boxes, scores, area = boxes[keep], scores[keep], area[keep]
    if max_rel_area is not None:
        rel = area / np.float32(width * height)
        keep = rel < np.float32(max_rel_area)
        boxes, scores = boxes[keep], scores[keep]

    order = nms(boxes, scores, nms_iou_single if single else nms_iou)
    kept = [
        [
            float(boxes[i, 0]) / width,
            float(boxes[i, 1]) / height,
            float(boxes[i, 2]) / width,
            float(boxes[i, 3]) / height,
            float(scores[i]),
        ]
        for i in order
    ]

    # Drop boxes that contain another box (remove_inside=False)
    if drop_containers:
        kept = [
            b
            for k, b in enumerate(kept)
            if not any(bx.inside(o, b) for m, o in enumerate(kept) if m != k)
        ]

    # A single object keeps the boxes close to the best score
    if single and kept:
        best = max(b[4] for b in kept)
        kept = [b for b in kept if best - b[4] < HIGH_CONF_TOLERANCE]
    if len(kept) > topk:
        kept = sorted(kept, key=lambda b: b[4], reverse=True)[:topk]
    return kept


def children(raw: dict, size: list, request: dict, entities: list, parent_box, controls: dict):
    """The regions detection found: per entity ``[id, label, text]`` in order, its kept boxes
    as regions (``id`` numbered ``_0``, ``_1``, ... when there are several) in image
    coordinates."""
    width, height = size
    max_rel_area = None if parent_box is None else MAX_REL_AREA_ENTITY
    per_text: dict[int, tuple[list, list]] = {}
    for box, score, label in zip(raw["boxes"], raw["scores"], raw["labels"]):
        per_text.setdefault(label, ([], []))
        per_text[label][0].append(box)
        per_text[label][1].append(score)

    found = {}
    for label, (raw_boxes, raw_scores) in per_text.items():
        entity_id, entity_label, text = entities[label]
        kept = select(
            raw_boxes,
            raw_scores,
            width,
            height,
            entity_label == "single",
            max_rel_area,
            controls["max_instances_per_text"],
            controls["box_score"],
            controls["nms_iou"],
            controls["nms_iou_single"],
            controls["drop_containers"],
        )
        if kept:
            found[text] = kept

    out = []
    for entity_id, _, text in entities:
        kept = found.get(text, [])
        for i, box in enumerate(kept):
            out.append(
                {
                    "id": entity_id if len(kept) == 1 else f"{entity_id}_{i}",
                    "text": text,
                    "box": bx.to_global(parent_box, box),
                }
            )
    return out
