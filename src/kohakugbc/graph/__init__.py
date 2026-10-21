"""Records and their resolution into GBC graphs.

``to_text`` is the resolution the program uses: the Rust implementation (``kohakugbc._rs``)
when it is built, else the Python reference; ``KOHAKUGBC_RESOLVE=python|rust`` picks one. The
choice is made once, at import.
"""

import json
import os

from kohakugbc.graph.digest import canonical, digest
from kohakugbc.graph.orders import stats
from kohakugbc.graph.records import from_node_info, record, ref
from kohakugbc.graph.reference import ResolutionError, resolve

try:
    from kohakugbc import _rs
except ImportError:
    _rs = None


def python_text(
    records: list[dict], img_path: str | None, img_size: list | None = None, threshold=0.85
) -> tuple[str | None, dict]:
    """The Python reference: the graph as JSON text (None without described records), its stats."""
    graph = resolve(records, img_path, img_size, threshold)
    text = None if graph is None else json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    return text, stats(graph)


def rust_text(
    records: list[dict], img_path: str | None, img_size: list | None = None, threshold=0.85
) -> tuple[str | None, dict]:
    try:
        return _rs.resolve(records, img_path, img_size, threshold)
    except ValueError as exc:
        raise ResolutionError(str(exc)) from None


def _select():
    match os.environ.get("KOHAKUGBC_RESOLVE", "rust" if _rs is not None else "python"):
        case "python":
            return python_text
        case "rust" if _rs is not None:
            return rust_text
        case choice:
            raise ImportError(f"KOHAKUGBC_RESOLVE={choice!r}: not available")


to_text = _select()

__all__ = [
    "ResolutionError",
    "canonical",
    "digest",
    "from_node_info",
    "python_text",
    "record",
    "ref",
    "resolve",
    "rust_text",
    "stats",
    "to_text",
]
