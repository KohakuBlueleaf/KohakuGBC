"""Triton mask box: the mask logits and their box in one pass, no logits written.

The instance projection is folded into the query side (``E = emb . W`` in fp32, ``QB = emb .
b``, made by the caller) and ``E`` carried as two 16-bit parts (``E_HI + E_LO``: fp32-accurate;
one 16-bit ``E`` moved box edges by 2 grid pixels on average, as the edges sit where the logits
cross the threshold). Program ``(b, q-block, pixel-tile)`` takes ``BQ`` queries against ``BP``
pixel embeddings ``[C, BP]``: the logits tile in fp32, kept where above the threshold's logit,
and the kept pixels' extreme rows / columns folded into the box with atomic min / max.
"""

import triton
import triton.language as tl


@triton.jit
def mask_box_fwd(
    E,
    ELO,
    QB,
    PIX,
    BOX,
    threshold,
    Q,
    C,
    HW,
    width,
    seb,
    seq,
    spb,
    spc,
    sbb,
    sbq,
    BQ: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
):
    b = tl.program_id(0)
    qs = tl.program_id(1) * BQ + tl.arange(0, BQ)
    ps = tl.program_id(2) * BP + tl.arange(0, BP)
    in_q = qs < Q
    in_p = ps < HW
    acc = tl.zeros((BQ, BP), tl.float32)
    for c0 in range(0, C, BC):
        c = c0 + tl.arange(0, BC)
        offs = b * seb + qs[:, None] * seq + c[None, :]
        e = tl.load(E + offs, mask=in_q[:, None], other=0.0)
        e_lo = tl.load(ELO + offs, mask=in_q[:, None], other=0.0)
        p = tl.load(PIX + b * spb + c[:, None] * spc + ps[None, :], mask=in_p[None, :], other=0.0)
        acc = tl.dot(e, p, acc)
        acc = tl.dot(e_lo, p, acc)
    acc += tl.load(QB + b * Q + qs, mask=in_q, other=0.0)[:, None]
    keep = (acc > threshold) & in_q[:, None] & in_p[None, :]
    rows = ps // width
    cols = ps % width
    big = 1 << 30
    top = tl.min(tl.where(keep, rows[None, :], big), axis=1)
    left = tl.min(tl.where(keep, cols[None, :], big), axis=1)
    bottom = tl.max(tl.where(keep, rows[None, :] + 1, 0), axis=1)
    right = tl.max(tl.where(keep, cols[None, :] + 1, 0), axis=1)
    any_kept = tl.max(keep.to(tl.int32), axis=1) > 0
    box = BOX + b * sbb + qs * sbq
    tl.atomic_min(box + 0, left, mask=in_q & any_kept)
    tl.atomic_min(box + 1, top, mask=in_q & any_kept)
    tl.atomic_max(box + 2, right, mask=in_q & any_kept)
    tl.atomic_max(box + 3, bottom, mask=in_q & any_kept)
