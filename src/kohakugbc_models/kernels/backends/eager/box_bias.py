"""SAM 3's decoder box relative-position bias, eager, as transformers computes it (every op in
the boxes' dtype).

``box_bias(boxes, y_w1, y_b1, y_w2, y_b2, x_w1, x_b1, x_w2, x_b2, height, width)``: query boxes
``[B, Q, 4]`` (cx, cy, w, h, 0..1) to the separable bias tables ``[B, heads, 1 + Q, height]``
and ``[B, heads, 1 + Q, width]`` (row 0, the presence token, zero; contiguous). Per axis the
deltas of every grid coordinate (``g / size``) to the box's two edges, ``* 8``, log-scaled
(``sign * log2(|d| + 1) / 3``), through the axis' MLP (``w1 [hidden, 2]``, ``w2 [heads,
hidden]``): ``w2 . relu(w1 . d + b1) + b2``.
"""

import math

import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import registry


def axis_table(edges: torch.Tensor, mlp: tuple, size: int) -> torch.Tensor:
    """``edges`` ``[B, Q, 2]`` (one axis' low / high edge) to ``[B, heads, 1 + Q, size]``."""
    w1, b1, w2, b2 = mlp
    coords = torch.arange(0, size, device=edges.device, dtype=edges.dtype) / size
    d = coords.view(1, 1, -1, 1) - edges[:, :, None, :]
    d = d * 8
    d = torch.sign(d) * torch.log2(torch.abs(d) + 1.0) / math.log2(8)
    table = F.linear(F.relu(F.linear(d, w1, b1)), w2, b2)
    return F.pad(table.permute(0, 3, 1, 2), (0, 0, 1, 0), mode="constant", value=0).contiguous()


def box_bias(boxes, y_w1, y_b1, y_w2, y_b2, x_w1, x_b1, x_w2, x_b2, height, width) -> tuple:
    cx, cy, w, h = boxes.unbind(-1)
    ys = torch.stack([cy - 0.5 * h, cy + 0.5 * h], dim=-1)
    xs = torch.stack([cx - 0.5 * w, cx + 0.5 * w], dim=-1)
    y = axis_table(ys, (y_w1, y_b1, y_w2, y_b2), height)
    return y, axis_table(xs, (x_w1, x_b1, x_w2, x_b2), width)


registry.register("box_bias", "eager", box_bias)
