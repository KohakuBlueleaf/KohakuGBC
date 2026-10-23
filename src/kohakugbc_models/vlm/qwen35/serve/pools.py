"""What the engine keeps per sequence on the device, and the prefix snapshots.

* **State slots**: per DeltaNet layer, one recurrent state ``[value_heads, key_dim, value_dim]``
  and one convolution state ``[channels, width - 1]`` per slot. Slots serve running sequences
  and snapshots alike; the last slot is scratch (padded rows of a captured graph write there).
* **KV pages**: per attention layer, ``[pages, page, kv_heads, head_dim]`` keys and values;
  page 0 is scratch. Pages are reference counted: a snapshot's full pages are shared by every
  sequence resumed from it.
* **Snapshots**: a prefix (its key: the prefix tokens and the images among them) with a slot
  holding the states after it, its full pages, and a private copy of its last partial page.
  A sequence resumed from one copies the states into its slot, shares the full pages and copies
  the partial one, so a snapshot can go at any time: the least recently used goes when the
  snapshots are full or slots or pages run short.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from kohakugbc_models.vlm.qwen35.model.shape import Shape


@dataclass
class Snapshot:
    key: bytes
    length: int
    slot: int
    pages: list[int]  # full pages, shared
    tail: int | None  # private copy of the last partial page


@dataclass
class Plan:
    """How many slots and pages fit, and what they take."""

    slots: int
    pages: int
    page: int
    state_bytes: int
    kv_bytes: int
    reasons: list[str] = field(default_factory=list)


def plan(
    shape: Shape,
    budget: int,
    seqs: int,
    snapshots: int,
    page: int,
    state_dtype,
    dtype,
    min_tokens: int,
) -> Plan:
    """Slots for ``seqs`` sequences, ``snapshots`` snapshots and the scratch slot; every byte of
    ``budget`` left after them as KV pages (at least ``min_tokens`` worth)."""
    st = torch.empty((), dtype=state_dtype).element_size()
    it = torch.empty((), dtype=dtype).element_size()
    slots = seqs + snapshots + 1
    state = slots * (shape.state_bytes(st) + shape.conv_bytes(it))
    per_page = page * shape.kv_bytes_per_token(it)
    pages = (budget - state) // per_page if budget > state else 0
    out = Plan(slots, int(pages), page, state, int(pages) * per_page)
    if pages * page < min_tokens:
        out.reasons.append(
            f"KV pages for {pages * page} tokens after {state / 2**30:.2f} GiB of state slots; "
            f"need {min_tokens} (raise memory, lower max_seqs, snapshots or max_model_len)"
        )
    return out


class Pools:
    def __init__(self, shape: Shape, p: Plan, device, state_dtype, dtype, max_snapshots: int):
        nd, na = len(shape.delta_layers), len(shape.attention_layers)
        self.shape, self.page = shape, p.page
        self.max_snapshots = max_snapshots
        self.state = torch.zeros(
            nd,
            p.slots,
            shape.value_heads,
            shape.key_dim,
            shape.value_dim,
            dtype=state_dtype,
            device=device,
        )
        self.conv = torch.zeros(
            nd, p.slots, shape.conv_channels, shape.conv_width - 1, dtype=dtype, device=device
        )
        self.keys = torch.zeros(
            na, p.pages, p.page, shape.kv_heads, shape.head_dim, dtype=dtype, device=device
        )
        self.values = torch.zeros_like(self.keys)
        self.scratch_slot = p.slots - 1
        self.free_slots = list(range(p.slots - 2, -1, -1))
        self.free_pages = list(range(p.pages - 1, 0, -1))
        self.refs = [0] * p.pages
        self.snapshots: OrderedDict[bytes, Snapshot] = OrderedDict()
        self.counts = {"snapshots_taken": 0, "snapshot_hits": 0, "snapshots_evicted": 0}

    # ---- slots and pages -------------------------------------------------------------------------

    def take_slot(self) -> int | None:
        if not self.free_slots and not self._evict():
            return None
        return self.free_slots.pop()

    def clear_slot(self, slot: int) -> None:
        self.state[:, slot].zero_()
        self.conv[:, slot].zero_()

    def give_slot(self, slot: int) -> None:
        self.free_slots.append(slot)

    def take_pages(self, n: int) -> list[int] | None:
        while len(self.free_pages) < n:
            if not self._evict():
                return None
        out = [self.free_pages.pop() for _ in range(n)]
        for p in out:
            self.refs[p] = 1
        return out

    def share(self, pages: list[int]) -> None:
        for p in pages:
            self.refs[p] += 1

    def give_pages(self, pages: list[int]) -> None:
        for p in pages:
            self.refs[p] -= 1
            if self.refs[p] == 0:
                self.free_pages.append(p)

    def pages_free(self) -> int:
        """Pages free now or held by snapshots alone."""
        held = 0
        for s in self.snapshots.values():
            held += sum(1 for p in s.pages if self.refs[p] == 1) + (s.tail is not None)
        return len(self.free_pages) + held

    def copy_page(self, src: int, dst: int) -> None:
        self.keys[:, dst] = self.keys[:, src]
        self.values[:, dst] = self.values[:, src]

    # ---- snapshots -------------------------------------------------------------------------

    def find(self, key: bytes) -> Snapshot | None:
        snap = self.snapshots.get(key)
        if snap is not None:
            self.snapshots.move_to_end(key)
        return snap

    def take_snapshot(self, key: bytes, length: int, slot: int, pages: list[int]) -> bool:
        """The states of ``slot`` and the first ``length`` tokens of ``pages`` as the snapshot of
        ``key``; False when no slot or page is free for it."""
        if key in self.snapshots:
            return True
        if self.max_snapshots <= 0:
            return False
        while len(self.snapshots) >= self.max_snapshots:
            self._evict()
        full, rest = divmod(length, self.page)
        snap_slot = self.take_slot()
        if snap_slot is None:
            return False
        tail = None
        if rest:
            got = self.take_pages(1)
            if got is None:
                self.give_slot(snap_slot)
                return False
            tail = got[0]
            self.copy_page(pages[full], tail)
        self.state[:, snap_slot] = self.state[:, slot]
        self.conv[:, snap_slot] = self.conv[:, slot]
        shared = pages[:full]
        self.share(shared)
        self.snapshots[key] = Snapshot(key, length, snap_slot, shared, tail)
        self.counts["snapshots_taken"] += 1
        return True

    def resume(self, snap: Snapshot, slot: int, pages: list[int]) -> list[int]:
        """``slot`` holds the snapshot's states; the page table starts with its shared pages and a
        copy of its partial page (in ``pages[0]``, which the caller allocated); returns the
        sequence's page table with ``pages`` after the shared ones."""
        self.state[:, slot] = self.state[:, snap.slot]
        self.conv[:, slot] = self.conv[:, snap.slot]
        self.share(snap.pages)
        if snap.tail is not None:
            self.copy_page(snap.tail, pages[0])
        self.counts["snapshot_hits"] += 1
        return list(snap.pages) + pages

    def _evict(self) -> bool:
        if not self.snapshots:
            return False
        _, snap = self.snapshots.popitem(last=False)
        self.give_pages(snap.pages)
        if snap.tail is not None:
            self.give_pages([snap.tail])
        self.free_slots.append(snap.slot)
        self.counts["snapshots_evicted"] += 1
        return True
