"""GBC's text rules: singular forms, phrase positions, which names are detected, same object.

Ports of ml-gbc's ``gbc.texts`` and ``io_unit.find_ref_poss``, and the PoC's same-object rule.
"""

import re
from functools import lru_cache

import numpy as np
from hbutils.string.plural import singular_form

QUANTIFIERS = {
    "a", "an", "some", "many", "several", "few", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}  # fmt: skip
POSITIONS = {"right", "left", "top", "bottom", "front", "back", "side"}


@lru_cache(maxsize=65536)
def singular(text: str) -> str:
    """ml-gbc's ``plural_to_singular``: a leading quantifier dropped, singular, lower case."""
    if " " in text:
        parts = text.split()
        if parts[0].lower() in QUANTIFIERS:
            text = " ".join(parts[1:])
    return singular_form(text).lower()


def ref_count(texts: list[str], phrase: str) -> int:
    """How often the phrase occurs as whole words in the texts (ml-gbc's ``find_ref_poss``)."""
    pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    return sum(len(pattern.findall(t)) for t in texts)


def detectable(text: str) -> bool:
    """ml-gbc's ``suitable_for_detection`` without a classifier."""
    return text not in POSITIONS and not text.isdigit() and "_" not in text


def same_object(a: str, b: str, va: np.ndarray, vb: np.ndarray, threshold: float) -> bool:
    """The PoC's rule: equal singular forms, or embeddings at least ``threshold`` alike."""
    if singular(a) == singular(b):
        return True
    return float(va @ vb) >= threshold
