"""Triton box relative-position bias: boxes to one axis' bias table in one pass, the MLP's
hidden layer never leaving the chip.

Program ``(b, q)`` takes query ``q`` of batch row ``b`` and every grid coordinate of the axis
(``BG`` rows, ``BG`` >= the grid size): the two edge deltas and their log scaling in fp32, then
the hidden layer ``BK`` units at a time (fp32, ReLU, rounded to the weights' dtype) into the
output layer's dot (``NH``: the heads padded to 16 for the MMA). Row 0 of the table (the
presence token) is left to the caller's zeros.
"""

import triton
import triton.language as tl

INV_LOG2_8 = tl.constexpr(1.0 / 3.0)


@triton.jit
def axis_table(
    BOXES,
    W1,
    B1,
    W2,
    B2,
    OUT,
    sbb,
    sbq,
    sob,
    soh,
    soq,
    size,
    AXIS: tl.constexpr,
    HID: tl.constexpr,
    HEADS: tl.constexpr,
    NH: tl.constexpr,
    BG: tl.constexpr,
    BK: tl.constexpr,
):
    b = tl.program_id(0)
    q = tl.program_id(1)
    g = tl.arange(0, BG)
    in_g = g < size
    centre = tl.load(BOXES + b * sbb + q * sbq + AXIS).to(tl.float32)
    extent = tl.load(BOXES + b * sbb + q * sbq + 2 + AXIS).to(tl.float32)
    coord = g.to(tl.float32) / size
    d0 = (coord - (centre - 0.5 * extent)) * 8.0
    d1 = (coord - (centre + 0.5 * extent)) * 8.0
    e0 = tl.where(d0 >= 0, 1.0, -1.0) * tl.log2(tl.abs(d0) + 1.0) * INV_LOG2_8
    e1 = tl.where(d1 >= 0, 1.0, -1.0) * tl.log2(tl.abs(d1) + 1.0) * INV_LOG2_8
    e0 = tl.where(d0 == 0, 0.0, e0)
    e1 = tl.where(d1 == 0, 0.0, e1)
    heads = tl.arange(0, NH)
    acc = tl.zeros((BG, NH), tl.float32)
    for k0 in range(0, HID, BK):
        k = k0 + tl.arange(0, BK)
        w10 = tl.load(W1 + k * 2).to(tl.float32)
        w11 = tl.load(W1 + k * 2 + 1).to(tl.float32)
        b1 = tl.load(B1 + k).to(tl.float32)
        hidden = e0[:, None] * w10[None, :] + e1[:, None] * w11[None, :] + b1[None, :]
        hidden = tl.maximum(hidden, 0.0).to(W2.dtype.element_ty)
        w2 = tl.load(
            W2 + heads[None, :] * HID + k[:, None], mask=(heads < HEADS)[None, :], other=0.0
        )
        acc = tl.dot(hidden, w2, acc)
    acc += tl.load(B2 + heads, mask=heads < HEADS, other=0.0).to(tl.float32)[None, :]
    out = OUT + b * sob + heads[None, :] * soh + (1 + q) * soq + g[:, None]
    tl.store(out, acc.to(OUT.dtype.element_ty), mask=in_g[:, None] & (heads < HEADS)[None, :])
