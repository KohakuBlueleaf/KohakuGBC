"""Deterministic draws: GBC's randomised choices (how many relations to ask for, which phrase
names a set of children, the composition hints) drawn from a generator seeded by what is being
asked, so a job gives the same requests on every run and on every machine."""

import hashlib
import random


def rng(*parts) -> random.Random:
    """A generator seeded by the parts (stringified, joined), e.g. ``rng("relation", img, vid)``."""
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(data).digest()[:8], "big"))
