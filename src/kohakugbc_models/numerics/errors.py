"""Error profiles against a ruler: how a tensor computed in some precision differs from the
same quantity computed in fp64.

``profile(got, ruler)`` gives the cosine similarity, the absolute and relative error as mean /
median / max, and the worst elements with their ruler magnitude, so a reviewer can tell a real
error from a large relative error on a near-zero true value. No verdict is attached: the
profiles are compared by hand against the same-precision PyTorch baseline.
"""

import torch


def profile(got: torch.Tensor, ruler: torch.Tensor, worst: int = 5, tiny: float = 1e-6) -> dict:
    g = got.detach().to(torch.float64).flatten()
    r = ruler.detach().to(torch.float64).flatten()
    if g.numel() == 0:
        return {"n": 0}
    finite = torch.isfinite(g)
    diff = (g - r).abs()
    scale = r.abs()
    rel = diff / scale.clamp(min=tiny)
    cos = torch.nn.functional.cosine_similarity(g[None], r[None]).item()
    order = diff.argsort(descending=True)[:worst]
    rel_order = rel.argsort(descending=True)[:worst]

    def stats(x: torch.Tensor) -> dict:
        return {"mean": x.mean().item(), "median": x.median().item(), "max": x.max().item()}

    return {
        "n": g.numel(),
        "nonfinite": int((~finite).sum()),
        "cosine": cos,
        "abs": stats(diff),
        "rel": stats(rel),
        "ruler_abs": stats(scale),
        "worst_abs": [{"index": int(i), "got": g[i].item(), "ruler": r[i].item()} for i in order],
        "worst_rel": [
            {"index": int(i), "got": g[i].item(), "ruler": r[i].item()} for i in rel_order
        ],
    }


def merge(profiles: list[dict]) -> dict:
    """One profile per field over several tensors' profiles (means weighted by size; worst kept)."""
    rows = [p for p in profiles if p.get("n")]
    if not rows:
        return {"n": 0}
    n = sum(p["n"] for p in rows)
    out = {"n": n, "nonfinite": sum(p["nonfinite"] for p in rows)}
    out["cosine_min"] = min(p["cosine"] for p in rows)
    for field in ("abs", "rel", "ruler_abs"):
        out[field] = {
            "mean": sum(p[field]["mean"] * p["n"] for p in rows) / n,
            "median_of_medians": sorted(p[field]["median"] for p in rows)[len(rows) // 2],
            "max": max(p[field]["max"] for p in rows),
        }
    out["worst_abs"] = max(
        (w for p in rows for w in p["worst_abs"]), key=lambda w: abs(w["got"] - w["ruler"])
    )
    return out
