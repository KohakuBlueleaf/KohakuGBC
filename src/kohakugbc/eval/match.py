"""The correspondence between a predicted graph and the ground truth.

A correspondence ``m`` maps predicted vertices one-to-one to ground-truth vertices (the image
to the image); a pair is allowed when its IoU reaches ``tau`` and the types agree. A track's
score is linear in the matched pairs, edges and relations, so the correspondence that maximizes
it is an integer program, solved exactly (HiGHS):

- ``x[u, v]``: u matched to v; at most one per vertex on each side.
- ``y[e, f]``: predicted edge e = (a, b) mapped onto ground-truth edge f = (c, d);
  ``y <= x[b, d]`` and ``y <= x[a, c]`` (nothing when a and c are the image).
- ``z[r, s]``: predicted relation r mapped onto ground-truth relation s, ``|r| = |s|``;
  ``z <= sum_{v in s} x[u, v]`` for every member u of r (with one-to-one x, m(r) = s).

The objective is ``sum wv[u, v] x + we sum y + sum wr[r, s] z``, with ties broken toward the
larger summed IoU. Only boxes, types and structure enter it, never vertex ids or order.
``m[u]`` is the ground-truth index of predicted vertex u, or -1.
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_array

from kohakugbc.eval.spec import IMAGE, Spec

TIE = 1e-7


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def allowed(pred: Spec, gt: Spec, ious: np.ndarray, tau: float) -> np.ndarray:
    ok = ious >= tau
    if ok.size:
        ok &= np.asarray(pred.types, object)[:, None] == np.asarray(gt.types, object)[None, :]
    return ok


def solve(
    pred: Spec,
    gt: Spec,
    ok: np.ndarray,
    ious: np.ndarray,
    wv: np.ndarray,
    we: float,
    wr: np.ndarray,
) -> np.ndarray:
    m = np.full(len(pred), -1)
    pairs = [(int(u), int(v)) for u, v in zip(*np.nonzero(ok))]
    if not pairs:
        return m
    # No vertex has a choice: every weight is positive, so matching every pair is best
    if (ok.sum(0) <= 1).all() and (ok.sum(1) <= 1).all():
        for u, v in pairs:
            m[u] = v
        return m
    xi = {p: k for k, p in enumerate(pairs)}
    tie = TIE / len(pairs)
    cost = [-(wv[u, v] + tie * ious[u, v]) for u, v in pairs]
    rows, ub = [], []
    for side in (0, 1):
        groups: dict = {}
        for k, p in enumerate(pairs):
            groups.setdefault(p[side], []).append(k)
        for ks in groups.values():
            if len(ks) > 1:
                _row(rows, ub, [(k, 1.0) for k in ks], 1.0)

    by_u: dict = {}
    for u, v in pairs:
        by_u.setdefault(u, []).append(v)
    parents: dict = {}
    for c, d in gt.edges:
        parents.setdefault(d, []).append(c)
    if we > 0:
        for a, b in pred.edges:
            for d in by_u.get(b, ()):
                for c in parents.get(d, ()):
                    if (a == IMAGE) != (c == IMAGE):
                        continue
                    if a != IMAGE and (a, c) not in xi:
                        continue
                    y = len(cost)
                    cost.append(-we)
                    _row(rows, ub, [(y, 1.0), (xi[(b, d)], -1.0)], 0.0)
                    if a != IMAGE:
                        _row(rows, ub, [(y, 1.0), (xi[(a, c)], -1.0)], 0.0)

    for ri, r in enumerate(pred.relations):
        for si, s in enumerate(gt.relations):
            if len(r) != len(s) or wr[ri, si] <= 0:
                continue
            members = [[xi[(u, v)] for v in s if (u, v) in xi] for u in r]
            if not all(members):
                continue
            z = len(cost)
            cost.append(-wr[ri, si])
            for ks in members:
                _row(rows, ub, [(z, 1.0)] + [(k, -1.0) for k in ks], 0.0)

    n = len(cost)
    cons = []
    if ub:
        r, c, val = zip(*rows)
        a = csr_array((val, (r, c)), shape=(len(ub), n))
        cons.append(LinearConstraint(a, -np.inf, np.asarray(ub)))
    res = milp(
        np.asarray(cost),
        constraints=cons,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},
    )
    if res.x is None:
        raise RuntimeError(f"the correspondence program failed: {res.message}")
    for k, (u, v) in enumerate(pairs):
        if res.x[k] > 0.5:
            m[u] = v
    return m


def _row(rows: list, ub: list, entries: list, bound: float) -> None:
    r = len(ub)
    rows.extend((r, c, val) for c, val in entries)
    ub.append(bound)
