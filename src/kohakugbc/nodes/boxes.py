"""Boxes as GBC uses them: ``[left, top, right, bottom, confidence]`` relative to the image.

Ports of ml-gbc's ``Bbox`` arithmetic, operation for operation, so results agree to the bit.
"""


def area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap_area(a, b) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def overlapped(a, b, threshold: float) -> bool:
    """Both boxes covered above the threshold by their intersection (``is_overlapped``)."""
    inter = overlap_area(a, b)
    area_a, area_b = area(a), area(b)
    ratio_a = inter / area_a if area_a > 0 else 1
    ratio_b = inter / area_b if area_b > 0 else 1
    return ratio_a > threshold and ratio_b > threshold


def inside(a, b) -> bool:
    """``a`` lies within ``b`` (``is_inside``)."""
    return a[0] >= b[0] and a[2] <= b[2] and a[1] >= b[1] and a[3] <= b[3]


def union(boxes: list) -> list:
    left, top, right, bottom = boxes[0][:4]
    for b in boxes[1:]:
        left, top = min(left, b[0]), min(top, b[1])
        right, bottom = max(right, b[2]), max(bottom, b[3])
    return [left, top, right, bottom, None]


def to_global(outer, inner) -> list:
    """A box relative to ``outer`` as a box relative to the image (``convert_to_global_bbox``)."""
    if outer is None:
        return list(inner)
    width, height = outer[2] - outer[0], outer[3] - outer[1]
    return [
        outer[0] + inner[0] * width,
        outer[1] + inner[1] * height,
        outer[0] + inner[2] * width,
        outer[1] + inner[3] * height,
        inner[4],
    ]


def to_relative(outer, inner) -> list:
    """A box relative to the image as a box relative to ``outer`` (``convert_to_relative_bbox``)."""
    width, height = outer[2] - outer[0], outer[3] - outer[1]
    return [
        (inner[0] - outer[0]) / width,
        (inner[1] - outer[1]) / height,
        (inner[2] - outer[0]) / width,
        (inner[3] - outer[1]) / height,
        inner[4],
    ]


def pixels(box, width: int, height: int) -> tuple[int, int, int, int]:
    """The pixel window ``crop_bbox`` cuts: (left, top, right, bottom), truncated."""
    return (
        int(box[0] * width),
        int(box[1] * height),
        int(box[2] * width),
        int(box[3] * height),
    )
