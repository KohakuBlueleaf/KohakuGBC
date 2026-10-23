"""SAM 3's mask box (``mask`` box source), eager, as the mask decoder's tail and
``detect.boxes.mask_boxes`` compute it.

``mask_box(emb, pixels, weight, bias, threshold)``: the instance projection (1 x 1 conv
``weight [C, C, 1, 1]`` / ``bias``) of the pixel embeddings ``[B, C, H, W]``, the mask logits
``emb [B, Q, C]`` . instance ``[B, Q, H, W]``, binarized at ``sigmoid > threshold``; per query
the box of its kept pixels on the ``H x W`` grid, ``[B, Q, 4]`` int32 (left, top, right, bottom;
right / bottom one past the last kept column / row), and whether none was kept ``[B, Q]``.
"""

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry


def mask_box(emb, pixels, weight, bias, threshold: float) -> tuple:
    instance = F.conv2d(pixels, weight, bias)
    logits = torch.einsum("bqc,bchw->bqhw", emb, instance)
    keep = logits.float().sigmoid() > threshold
    rows, cols = keep.any(dim=3), keep.any(dim=2)
    height, width = keep.shape[2:]
    top = rows.int().argmax(dim=2)
    bottom = height - rows.flip(2).int().argmax(dim=2)
    left = cols.int().argmax(dim=2)
    right = width - cols.flip(2).int().argmax(dim=2)
    return torch.stack([left, top, right, bottom], dim=2).int(), ~rows.any(dim=2)


registry.register("mask_box", "eager", mask_box)
