"""The two tracks' scores.

At IoU threshold ``tau``, under a correspondence ``m`` (``match``):

- ``TP_V``: matched vertices.
- ``TP_E``: predicted edges (a, b) with (m(a), m(b)) a ground-truth edge (m(image) = image).
- ``TP_R``: predicted relations r with every member matched and m(r) a ground-truth relation.
- ``F_X = 2 TP_X / (|X_pred| + |X_gt|)``; a part empty on both sides is inactive and left
  out of every mean (a track with no active part scores 1).

Structure track (boxes, types, graph): ``S`` = mean of ``F_V, F_E, F_R``.

Full track (with text, ``text.Sims``): ``S_full = (S + T) / 2``, ``T`` = mean of ``F_N, F_D,
F_Q, F_C``, with similarity in place of a hit: ``F_N = 2 sum_{u matched} s_N(u, m(u)) /
(|V_pred| + |V_gt|)`` (names), ``F_D`` the same for descriptions, ``F_Q = 2 sum_{r matched}
s_Q(r, m(r)) / (|R_pred| + |R_gt|)`` (relation descriptions), ``F_C`` the image's descriptions
(counted 1 per side that has any).

Each track takes the correspondence that maximizes its own score. An image's score is the mean
over the thresholds; the dataset's is the mean over images. Diagnostics, under each track's
correspondence: per part, counts summed over images and their precision/recall/F1, per vertex
type too; vertex AP over the dataset (box confidences, COCO's greedy matching, 101 points).
"""

from dataclasses import asdict, dataclass

import numpy as np

from kohakugbc.eval.match import allowed, iou, solve
from kohakugbc.eval.spec import IMAGE, Spec

TAUS = tuple(round(0.5 + 0.05 * k, 2) for k in range(10))
STRUCTURE = ("vertex", "edge", "relation")
TEXT = ("names", "descs", "relation_descs", "image_descs")


@dataclass(frozen=True)
class Options:
    taus: tuple = TAUS


DEFAULT = Options()


def f1(tp: float, p: float, g: float) -> float:
    return 1.0 if p + g == 0 else 2 * tp / (p + g)


def ratio(a: float, b: float) -> float:
    return 1.0 if b == 0 else a / b


def _w(n: int, share: float) -> float:
    return 2 * share / n if n else 0.0


def mean_active(parts: list) -> float:
    """Mean F1 over the parts not empty on both sides; 1 when none is."""
    got = [f1(*c) for c in parts if c[1] + c[2] > 0]
    return float(np.mean(got)) if got else 1.0


def measure(pred: Spec, gt: Spec, m: np.ndarray, sims: dict | None) -> dict:
    """Counts of every part under ``m``: [true positives, predicted, ground truth]."""

    def to(k):
        return IMAGE if k == IMAGE else (int(m[k]) if m[k] >= 0 else None)

    pairs = [(u, int(m[u])) for u in range(len(pred)) if m[u] >= 0]
    mapped_e = {(to(a), to(b)) for a, b in pred.edges}
    gt_r = {s: k for k, s in enumerate(gt.relations)}
    hits_r = []
    for ri, r in enumerate(pred.relations):
        s = frozenset(to(u) for u in r)
        if None not in s and s in gt_r:
            hits_r.append((ri, gt_r[s]))
    nv, nr = (len(pred), len(gt)), (len(pred.relations), len(gt.relations))
    out = {
        "vertex": [len(pairs), *nv],
        "edge": [len(mapped_e & gt.edges), len(pred.edges), len(gt.edges)],
        "relation": [len(hits_r), *nr],
        "types": {},
    }
    for typ in sorted({t for t in pred.types + gt.types if t is not None}):
        tp = sum(gt.types[v] == typ for _, v in pairs)
        out["types"][typ] = [tp, pred.types.count(typ), gt.types.count(typ)]
    if sims is not None:
        out["names"] = [sum(sims["names"][u, v] for u, v in pairs), *nv]
        out["descs"] = [sum(sims["descs"][u, v] for u, v in pairs), *nv]
        out["relation_descs"] = [sum(sims["relation_descs"][r, s] for r, s in hits_r), *nr]
        out["image_descs"] = [sims["image_descs"], *sims["image_sides"]]
    return out


def structure_score(c: dict) -> float:
    return mean_active([c[k] for k in STRUCTURE])


def full_score(c: dict) -> float:
    return (structure_score(c) + mean_active([c[k] for k in TEXT])) / 2


def correspond(pred: Spec, gt: Spec, ious: np.ndarray, tau: float, sims=None) -> np.ndarray:
    """The correspondence maximizing the structure score, or the full score with ``sims``."""
    ok = allowed(pred, gt, ious, tau)
    if not ok.any():
        return np.full(len(pred), -1)
    nv = len(pred) + len(gt)
    ne = len(pred.edges) + len(gt.edges)
    nr = len(pred.relations) + len(gt.relations)
    share = 1 / sum(n > 0 for n in (nv, ne, nr))
    wv = np.full(ok.shape, _w(nv, share))
    wr = np.full((len(pred.relations), len(gt.relations)), _w(nr, share))
    we = _w(ne, share)
    if sims is not None:
        # names and descriptions are active with any vertex, relation descriptions with any
        # relation, the image's with a description on either side; C does not depend on m
        k = 2 * (nv > 0) + (nr > 0) + (sum(sims["image_sides"]) > 0)
        wv, wr, we = wv / 2, wr / 2, we / 2
        wv = wv + _w(nv, 1 / (2 * k)) * (sims["names"] + sims["descs"])
        wr = wr + _w(nr, 1 / (2 * k)) * sims["relation_descs"]
    return solve(pred, gt, ok, ious, wv, we, wr)


def _add(acc: dict, c: dict) -> None:
    for k, x in c.items():
        if k == "types":
            types = acc.setdefault("types", {})
            for t, y in x.items():
                types[t] = [a + b for a, b in zip(types.get(t, [0, 0, 0]), y)]
        else:
            acc[k] = [a + b for a, b in zip(acc.get(k, [0] * len(x)), x)]


def _prf(tp: float, p: float, g: float) -> dict:
    return {"precision": ratio(tp, p), "recall": ratio(tp, g), "f1": f1(tp, p, g)}


def _summary(acc: dict) -> dict:
    out = {k: _prf(*x) for k, x in acc.items() if isinstance(x, list)}
    out["types"] = {t: _prf(*x) for t, x in acc.get("types", {}).items()}
    out["counts"] = acc
    return out


def vertex_ap(items: list, tau: float) -> float | None:
    groups = sorted({t for _, g, _ in items for t in g.types if t is not None})
    aps = []
    for typ in groups:
        ranked, total = [], 0
        for p, g, ious in items:
            gi = [j for j in range(len(g)) if g.types[j] == typ]
            pi = [i for i in range(len(p)) if p.types[i] == typ]
            total += len(gi)
            used = set()
            for i in sorted(pi, key=lambda i: -p.scores[i]):
                best, bj = tau, -1
                for j in gi:
                    if j not in used and ious[i, j] >= best:
                        best, bj = ious[i, j], j
                if bj >= 0:
                    used.add(bj)
                ranked.append((-p.scores[i], bj >= 0))
        if not total:
            continue
        ranked.sort(key=lambda x: x[0])
        tp = np.cumsum([h for _, h in ranked]) if ranked else np.zeros(0)
        prec = tp / np.arange(1, len(tp) + 1) if len(tp) else tp
        rec = tp / total
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        at = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        aps.append(float(np.mean([prec[k] if k < len(prec) else 0.0 for k in at])))
    return float(np.mean(aps)) if aps else None


def evaluate(pairs, opt: Options = DEFAULT, sims=None) -> tuple[dict, list]:
    """``pairs``: (key, predicted Spec, ground-truth Spec); ``sims``: a ``text.Sims`` for the
    full track, else the structure track only. Returns the report and the per-image rows."""
    tracks = ("structure",) if sims is None else ("structure", "full")
    scorer = {"structure": structure_score, "full": full_score}
    items, rows = [], []
    acc = {t: {tau: {} for tau in opt.taus} for t in tracks}
    for key, pred, gt in pairs:
        ious = iou(pred.boxes, gt.boxes)
        items.append((pred, gt, ious))
        text = None if sims is None else sims(pred, gt)
        row = {"key": key}
        for track in tracks:
            per = {}
            for tau in opt.taus:
                m = correspond(pred, gt, ious, tau, text if track == "full" else None)
                c = measure(pred, gt, m, text if track == "full" else None)
                _add(acc[track][tau], c)
                per[str(tau)] = {"score": scorer[track](c), "counts": c}
            row[track] = {"score": float(np.mean([p["score"] for p in per.values()])), "taus": per}
        rows.append(row)

    report = {"options": asdict(opt), "images": len(rows)}
    for track in tracks:
        per_tau = {}
        for tau in opt.taus:
            s = _summary(acc[track][tau])
            s["score"] = float(np.mean([r[track]["taus"][str(tau)]["score"] for r in rows]))
            if track == "structure":
                s["vertex_ap"] = vertex_ap(items, tau)
            per_tau[str(tau)] = s
        report[track] = {
            "score": float(np.mean([r[track]["score"] for r in rows])) if rows else 0.0,
            "taus": per_tau,
        }
    return report, rows
