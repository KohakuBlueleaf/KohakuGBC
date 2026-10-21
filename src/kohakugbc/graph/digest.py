"""The canonical text of a graph and its digest: how two resolutions are compared."""

import hashlib
import json


def canonical(graph: dict | str | None) -> str:
    """Compact JSON in the graph's own key order; a JSON text is parsed first, so texts that
    differ only in number or escape spelling are equal."""
    if isinstance(graph, str):
        graph = json.loads(graph)
    return json.dumps(graph, ensure_ascii=False, separators=(",", ":"))


def digest(graph: dict | str | None) -> str:
    return hashlib.sha256(canonical(graph).encode("utf-8")).hexdigest()
