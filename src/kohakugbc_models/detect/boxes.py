"""Boxes of binarized instance masks (SAM 3's mask box sources)."""

import torch

BOX_SOURCES = ("pred", "mask", "mask_full")


def mask_boxes(logits: torch.Tensor, size, threshold: float, full: bool, chunk: int) -> tuple:
    """Boxes ``[k, 4]`` (left, top, right, bottom in crop pixels) of ``k`` mask logits
    ``[k, h, w]`` binarized at ``threshold`` after the sigmoid, and which masks were empty.

    ``full`` resizes ``chunk`` masks at a time to the crop ``size`` (height, width) before the
    box; otherwise the box is taken on the ``h x w`` grid and scaled. A box covers whole pixels:
    right / bottom are one past the last kept column / row."""
    height, width = size
    boxes, empty = [], []
    for start in range(0, logits.shape[0], chunk if full else max(1, logits.shape[0])):
        part = logits[start : start + chunk] if full else logits
        if full:
            part = torch.nn.functional.interpolate(
                part.float().sigmoid().unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            part = part.float().sigmoid()
        keep = part > threshold
        rows, cols = keep.any(dim=2), keep.any(dim=1)
        grid_h, grid_w = keep.shape[1:]
        top = rows.int().argmax(dim=1)
        bottom = grid_h - rows.flip(1).int().argmax(dim=1)
        left = cols.int().argmax(dim=1)
        right = grid_w - cols.flip(1).int().argmax(dim=1)
        box = torch.stack([left, top, right, bottom], dim=1).float()
        box *= torch.tensor([width / grid_w, height / grid_h] * 2, device=box.device)
        boxes.append(box)
        empty.append(~rows.any(dim=1))
        if not full:
            break
    if not boxes:
        return logits.new_zeros(0, 4, dtype=torch.float32), logits.new_zeros(0, dtype=torch.bool)
    return torch.cat(boxes), torch.cat(empty)
