"""Orders, depths and counts of a resolved graph, as ml-gbc's ``GbcGraphFull`` computes them."""

from collections import Counter


def roots(graph: dict) -> list[dict]:
    """Image vertices first, then every other vertex without in-edges."""
    vertices = graph["vertices"]
    images = [v for v in vertices if v["label"] == "image"]
    others = [v for v in vertices if v["label"] != "image" and not v["in_edges"]]
    return images + others


def bfs(graph: dict) -> tuple[list[str], dict[str, int], int]:
    """The BFS order from the roots, each vertex's depth, and the graph depth."""
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    order, depths, seen = [], {}, set()
    current, depth = roots(graph), -1
    while current:
        depth += 1
        following = []
        for v in current:
            if v["vertex_id"] in seen:
                continue
            seen.add(v["vertex_id"])
            order.append(v["vertex_id"])
            depths[v["vertex_id"]] = depth
            following += [by_id[e["target"]] for e in v["out_edges"] if e["target"] not in seen]
        current = following
    return order, depths, depth


def dfs(graph: dict) -> tuple[list[str], list[str]]:
    """The DFS order from the roots and the topological order (reversed DFS finish order)."""
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    order, finished, seen = [], [], set()
    stack = [(v, False) for v in roots(graph)]
    while stack:
        v, done = stack.pop()
        if done:
            finished.append(v["vertex_id"])
            continue
        if v["vertex_id"] in seen:
            continue
        seen.add(v["vertex_id"])
        order.append(v["vertex_id"])
        stack.append((v, True))
        stack += [
            (by_id[e["target"]], False) for e in reversed(v["out_edges"]) if e["target"] not in seen
        ]
    return order, finished[::-1]


def stats(graph: dict | None) -> dict:
    """Counts of one graph: vertices by label, edges, captions, depth, roots, leaves."""
    if graph is None:
        return {"vertices": 0, "labels": {}, "edges": 0, "captions": 0, "depth": -1}
    vertices = graph["vertices"]
    _, _, depth = bfs(graph)
    return {
        "vertices": len(vertices),
        "labels": dict(sorted(Counter(v["label"] for v in vertices).items())),
        "edges": sum(len(v["out_edges"]) for v in vertices),
        "captions": sum(len(v["descs"]) for v in vertices),
        "depth": depth,
        "roots": len(roots(graph)),
        "leaves": sum(1 for v in vertices if not v["out_edges"]),
    }
