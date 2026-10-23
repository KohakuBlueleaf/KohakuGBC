"""SAM 3's mask boxes: the box of each binarized mask, on the model's grid scaled to the crop or
on the mask resized to the crop, chunk by chunk; an empty mask is flagged."""

import pytest

torch = pytest.importorskip("torch")

from kohakugbc_models.detect.boxes import mask_boxes  # noqa: E402

BIG = 30.0


def logits(grid: int, rects: list) -> torch.Tensor:
    """Mask logits ``[k, grid, grid]``: +BIG inside each (left, top, right, bottom) cell rect,
    -BIG elsewhere; None gives an empty mask."""
    out = torch.full((len(rects), grid, grid), -BIG)
    for k, rect in enumerate(rects):
        if rect is not None:
            left, top, right, bottom = rect
            out[k, top:bottom, left:right] = BIG
    return out


def test_grid_boxes_scale_to_the_crop():
    masks = logits(8, [(1, 2, 5, 7), (0, 0, 8, 8), None])
    boxes, empty = mask_boxes(masks, (40, 80), threshold=0.5, full=False, chunk=1)
    # Cells of 10 px wide, 5 px tall; right / bottom one past the last kept cell
    assert boxes[0].tolist() == [10.0, 10.0, 50.0, 35.0]
    assert boxes[1].tolist() == [0.0, 0.0, 80.0, 40.0]
    assert empty.tolist() == [False, False, True]


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_full_boxes_match_the_grid_boxes_at_an_integer_scale(chunk):
    # A sharp mask resized by an integer factor keeps its cell edges: the full-resolution box
    # equals the grid box, whatever the chunking
    masks = logits(8, [(1, 2, 5, 7), (3, 3, 4, 4), None, (0, 6, 8, 8)])
    grid, grid_empty = mask_boxes(masks, (32, 32), threshold=0.5, full=False, chunk=chunk)
    full, full_empty = mask_boxes(masks, (32, 32), threshold=0.5, full=True, chunk=chunk)
    assert torch.equal(grid_empty, full_empty)
    keep = ~grid_empty
    assert (full[keep] - grid[keep]).abs().max() <= 2.0
    # A box from the wrong mask (rows and columns swapped) is far off
    swapped, _ = mask_boxes(masks.transpose(1, 2), (32, 32), threshold=0.5, full=True, chunk=chunk)
    assert (swapped[0] - grid[0]).abs().max() >= 4.0


def test_no_masks():
    boxes, empty = mask_boxes(torch.zeros(0, 8, 8), (10, 10), threshold=0.5, full=True, chunk=4)
    assert boxes.shape == (0, 4) and empty.shape == (0,)
