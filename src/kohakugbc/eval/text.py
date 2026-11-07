"""Text similarity for the full track. Texts are never compared for equality: a vision-language
model names the same vertex differently from run to run, so two texts score the cosine of their
embeddings, rescaled against a baseline ``b``, the mean cosine of unrelated texts of the same
kind (texts of two different ground-truth images):

    s(a, b) = clip((cos(a, b) - b) / (1 - b), 0, 1)

- names (a vertex's aliases): the best pair over the two vertices' names.
- descriptions (a vertex's, a relation's, the image's): greedy F over sentences, as BERTScore
  does over tokens: P = mean over predicted sentences of the best ground-truth one, R the other
  way, F = 2PR / (P + R).
A side without texts scores 0.
"""

import numpy as np

FIELDS = ("names", "descs", "relation_descs", "image_descs")


class Embedder:
    """Unit-norm sentence embeddings from ``model`` (anything with sentence-transformers'
    ``encode``), cached per text."""

    def __init__(self, model):
        self.model = model
        self.cache: dict = {}

    def __call__(self, texts: list) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            vecs = self.model.encode(new, normalize_embeddings=True)
            self.cache.update(zip(new, np.asarray(vecs, float)))
        return np.stack([self.cache[t] for t in texts])

    def warm(self, specs, batch: int = 512) -> None:
        """Embed every text of ``specs`` ahead, in batches."""
        texts = list(dict.fromkeys(t for s in specs for f in FIELDS for t in _flat(s, f)))
        texts = [t for t in texts if t not in self.cache]
        for k in range(0, len(texts), batch):
            self(texts[k : k + batch])


def _flat(spec, name: str) -> list:
    got = getattr(spec, name)
    return list(got) if name == "image_descs" else [t for ts in got for t in ts]


def baselines(specs: list, embed, pairs: int = 2000, seed: int = 0) -> dict:
    """Per field, the mean cosine of a text of one ground-truth image and one of another."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in FIELDS:
        pools = [p for p in (_flat(s, name) for s in specs) if p]
        if len(pools) < 2:
            out[name] = 0.0
            continue
        a, b = [], []
        for _ in range(pairs):
            i, j = rng.choice(len(pools), 2, replace=False)
            a.append(pools[i][rng.integers(len(pools[i]))])
            b.append(pools[j][rng.integers(len(pools[j]))])
        out[name] = float(np.mean(np.sum(embed(a) * embed(b), axis=1)))
    return out


def rescale(cos, base: float):
    return np.clip((cos - base) / (1 - base), 0, 1)


def similarity(a, b, base: float, best: bool) -> float:
    if a is None or b is None:
        return 0.0
    s = rescale(a @ b.T, base)
    if best:
        return float(s.max())
    p, r = s.max(1).mean(), s.max(0).mean()
    return float(2 * p * r / (p + r)) if p + r else 0.0


class Sims:
    """The full track's text similarities of one image: vertex pairs (names, descriptions),
    relation pairs (descriptions), the image (descriptions, and which sides have any)."""

    def __init__(self, embed, base: dict):
        self.embed, self.base = embed, base

    def _vecs(self, groups: list) -> list:
        return [self.embed(g) if g else None for g in groups]

    def _matrix(self, a: list, b: list, name: str, best: bool) -> np.ndarray:
        va, vb = self._vecs(a), self._vecs(b)
        out = np.zeros((len(va), len(vb)))
        for i, x in enumerate(va):
            for j, y in enumerate(vb):
                out[i, j] = similarity(x, y, self.base[name], best)
        return out

    def __call__(self, pred, gt) -> dict:
        va, vb = self._vecs([pred.image_descs, gt.image_descs])
        image = similarity(va, vb, self.base["image_descs"], False)
        return {
            "names": self._matrix(pred.names, gt.names, "names", True),
            "descs": self._matrix(pred.descs, gt.descs, "descs", False),
            "relation_descs": self._matrix(
                pred.relation_descs, gt.relation_descs, "relation_descs", False
            ),
            "image_descs": image if pred.image_descs and gt.image_descs else 0.0,
            "image_sides": [int(bool(pred.image_descs)), int(bool(gt.image_descs))],
        }
