"""Composition hints and the annotated image of a composition query.

Ports of ml-gbc's ``MSTNode`` (a Euclidean minimum spanning tree over the parts' centres, walked
depth first from a root chosen by one of six criteria) and ``annotate_bboxes``. ml-gbc draws its
two random choices (the root criterion, the wording of diagonal directions) unseeded; here they
come from ``draw.rng("composition", image, entity_id)`` in the same order.
"""

import math
import random

import numpy as np
import supervision as sv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

CRITERIA = ("leftmost", "rightmost", "topmost", "bottommost", "middle_x", "middle_y")


def _centre(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _root(boxes: list, rng: random.Random) -> int:
    centres = np.array([_centre(b) for b in boxes])
    x, y = centres[:, 0], centres[:, 1]
    criteria = {
        "leftmost": np.argmin(x),
        "rightmost": np.argmax(x),
        "topmost": np.argmin(y),
        "bottommost": np.argmax(y),
        "middle_x": np.argsort(x)[len(x) // 2],
        "middle_y": np.argsort(y)[len(y) // 2],
    }
    return int(criteria[rng.choice(list(criteria.keys()))])


def _node_hint(text: str, rel, all_rel: list) -> str | None:
    cx, cy = _centre(rel)
    xs = [_centre(b)[0] for b in all_rel]
    ys = [_centre(b)[1] for b in all_rel]
    margin, low, high = 0.1, 0.4, 0.6
    horizontal = vertical = None
    if cx == min(xs) and rel[2] < 0.5:
        horizontal = "leftmost"
    elif cx == max(xs) and rel[0] > 0.5:
        horizontal = "rightmost"
    elif rel[0] < margin and rel[2] < 0.5:
        horizontal = "left"
    elif rel[2] > 1 - margin and rel[0] > 0.5:
        horizontal = "right"
    elif low < cx < high:
        horizontal = "center"
    if cy == min(ys) and rel[3] < 0.5:
        vertical = "topmost"
    elif cy == max(ys) and rel[1] > 0.5:
        vertical = "bottommost"
    elif rel[1] < margin and rel[3] < 0.5:
        vertical = "top"
    elif rel[3] > 1 - margin and rel[1] > 0.5:
        vertical = "bottom"
    elif low < cy < high:
        vertical = "center"

    corners = {
        "in the top left corner": [
            ("leftmost", "topmost"),
            ("leftmost", "top"),
            ("left", "topmost"),
        ],
        "in the top right corner": [
            ("rightmost", "topmost"),
            ("rightmost", "top"),
            ("right", "topmost"),
        ],
        "in the bottom left corner": [
            ("leftmost", "bottommost"),
            ("leftmost", "bottom"),
            ("left", "bottommost"),
        ],
        "in the bottom right corner": [
            ("rightmost", "bottommost"),
            ("rightmost", "bottom"),
            ("right", "bottommost"),
        ],
    }
    where = next((d for d, pairs in corners.items() if (horizontal, vertical) in pairs), None)
    if where is None:
        if horizontal == "leftmost":
            where = "on the left side"
        elif horizontal == "rightmost":
            where = "on the right side"
        elif vertical == "topmost":
            where = "at the top"
        elif vertical == "bottommost":
            where = "at the bottom"
        elif horizontal == "center" and vertical == "center":
            where = "in the center"
    return f"{text} is {where} of the composition" if where else None


def _edge_hint(parent_text: str, parent_box, child_text: str, child_box, rng) -> str:
    px, py = _centre(parent_box)
    cx, cy = _centre(child_box)
    angle = (math.atan2(cy - py, cx - px) * (180 / math.pi)) % 360
    diagonal = rng.random() < 0.4
    if 0 <= angle < 22.5 or 337.5 <= angle < 360:
        direction = "to the right of"
    elif 22.5 <= angle < 67.5:
        direction = "to the bottom-right of" if diagonal else "below and to the right of"
    elif 67.5 <= angle < 112.5:
        direction = "below"
    elif 112.5 <= angle < 157.5:
        direction = "to the bottom-left of" if diagonal else "below and to the left of"
    elif 157.5 <= angle < 202.5:
        direction = "to the left of"
    elif 202.5 <= angle < 247.5:
        direction = "to the top-left of" if diagonal else "above and to the left of"
    elif 247.5 <= angle < 292.5:
        direction = "above"
    else:
        direction = "to the top-right of" if diagonal else "above and to the right of"
    return f"{child_text} is {direction} {parent_text}."


def hints(texts: list[str], boxes: list, rel_boxes: list, rng: random.Random):
    """(hints, node texts in tree order) for parts with image boxes and crop-relative boxes."""
    centres = [list(_centre(b)) for b in boxes]
    mst = minimum_spanning_tree(csr_matrix(cdist(centres, centres, metric="euclidean")))
    mst = mst.toarray()
    mst = np.maximum(mst, mst.T)
    root = _root(boxes, rng)

    # The tree from the root, children in index order (ml-gbc's recursive DFS)
    visited: set[int] = set()
    children: dict[int, list[int]] = {}

    def build(i: int) -> None:
        visited.add(i)
        children[i] = []
        for j, weight in enumerate(mst[i]):
            if weight > 0 and j not in visited:
                build(j)
                children[i].append(j)

    build(root)

    # The hints in ml-gbc's order: a node's own hint first, then per child its edge hint and the
    # child's subtree; the direction wording draws once per edge, in that order
    def describe(i: int) -> tuple[list[str], list[str]]:
        out, names = [], [texts[i]]
        own = _node_hint(texts[i], rel_boxes[i], rel_boxes)
        if own:
            out.insert(0, own)
        for j in children[i]:
            out.append(_edge_hint(texts[i], boxes[i], texts[j], boxes[j], rng))
            sub, sub_names = describe(j)
            out += sub
            names += sub_names
        return out, names

    return describe(root)


def annotate(image: np.ndarray, rel_boxes: list) -> np.ndarray:
    """ml-gbc's ``annotate_bboxes``: green numbered boxes on the crop (supervision)."""
    height, width = image.shape[0], image.shape[1]
    ref = (width + height) / 2
    thickness = int(ref / 100)
    box_annotator = sv.BoxAnnotator(thickness=thickness, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator(
        text_thickness=int(thickness / 2),
        text_position=sv.Position.CENTER,
        color=sv.Color.GREEN,
        text_scale=ref / 500,
        text_padding=thickness,
        text_color=sv.Color.BLACK,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [(b[0] * width, b[1] * height, b[2] * width, b[3] * height) for b in rel_boxes]
        ),
        confidence=np.array([b[4] for b in rel_boxes]),
        class_id=np.zeros(len(rel_boxes), dtype=int),
    )
    out = box_annotator.annotate(image.copy(), detections)
    return label_annotator.annotate(
        out, detections, labels=[str(i) for i in range(1, len(rel_boxes) + 1)]
    )
