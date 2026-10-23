"""SAM 3's device memory plan: measured per card, stored, read by the engine.

A plan (``scripts/bench/detector/memory.py`` measures it) holds, for one card, dtype and set of
knobs: the resident bytes after load, the vision pass's working bytes as ``base + per_crop *
crops``, the fusion pass's as ``base + per_pair * pairs``, the bytes the graph pools (every pass
size up to the cap) and the text cache keep, and the allocator's reserve over its peak.
``estimate(crops, pairs)`` is the device bytes a detector with those caps reserves at its peak;
``fit(budget)`` the largest caps (pairs first, then crops, powers of two) whose estimate stays
within ``budget`` bytes.

Plans live in ``<kernel cache>/memory/<card key>/<name>.json``.
"""

import json
from dataclasses import asdict, dataclass

from kohakugbc_models.kernels.hardware import cache_dir, card_key


@dataclass
class Plan:
    card: str
    knobs: dict
    resident: int
    vision_base: int
    per_crop: int
    fusion_base: int
    per_pair: int
    kept: int
    reserve: int = 0

    def estimate(self, crops: int, pairs: int) -> int:
        vision = self.vision_base + self.per_crop * crops
        fusion = self.fusion_base + self.per_pair * pairs
        return self.resident + self.kept + max(vision, fusion) + self.reserve

    def fit(self, budget: int, max_crops: int = 4, max_pairs: int = 16) -> tuple | None:
        """``(crops, pairs)``: the largest caps within ``budget``, or None when even 1 / 1
        does not fit."""
        for pairs in (p for p in (64, 32, 16, 8, 4, 2, 1) if p <= max_pairs):
            for crops in (c for c in (4, 2, 1) if c <= max_crops):
                if self.estimate(crops, pairs) <= budget:
                    return crops, pairs
        return None


def path(device, name: str):
    return cache_dir() / "memory" / card_key(str(device)) / f"{name}.json"


def save(plan: Plan, device, name: str) -> None:
    target = path(device, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(plan), indent=1), encoding="utf-8")


def load(device, name: str) -> Plan | None:
    try:
        return Plan(**json.loads(path(device, name).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def plan_name(dtype: str, fp8: str | None, text_device: str | None, box_source: str) -> str:
    return f"sam3-{dtype}-fp8_{fp8 or 'none'}-text_{text_device or 'device'}-{box_source}"
