"""The node registry: node functions declared once, gathered into a ``NodeLibrary`` on demand.

Plain nodes are module-level functions (importable, so process workers can run them). Model
nodes (``chat``, ``detector``, ``embedder``) have no Python body of their own: a machine binds
them to engines, and the KIR reference interpreter gets them from a ``Models`` object.
"""

from typing import Protocol

from kohakunode import NodeLibrary

NODES: list[tuple] = []


def node(outputs: list[str] | None = None, **flags):
    def register(fn):
        NODES.append((fn, outputs, flags))
        return fn

    return register


class Models(Protocol):
    """What model nodes call when the program runs outside a machine."""

    def chat(self, request: dict) -> str: ...

    def detector(self, patch, request: dict) -> dict: ...

    def embedder(self, text: str): ...


def chat(request: dict) -> str:
    """A VLM reply to a request (``kohakugbc.nodes.specs.chat``); bound to the VLM lane."""
    raise NotImplementedError("chat runs on an engine; bind it, or build the library with models")


def detector(patch, request: dict) -> dict:
    """Raw boxes of the request's texts in the patch; bound to the detect lane."""
    raise NotImplementedError("detector runs on an engine")


def embedder(text: str):
    """The normalised embedding of a name; bound to the embed lane."""
    raise NotImplementedError("embedder runs on an engine")


def build(models: Models | None = None, name: str = "gbc") -> NodeLibrary:
    """The node library: every registered node; model nodes from ``models`` when given."""
    lib = NodeLibrary(name)
    for fn, outputs, flags in NODES:
        lib.node(outputs=outputs, **flags)(fn)
    model_nodes = (chat, detector, embedder) if models is None else _calling(models)
    for fn, output in zip(model_nodes, ("reply", "raw", "vec")):
        lib.node(outputs=[output])(fn)
    return lib


def _calling(models: Models) -> tuple:
    """Model nodes that call ``models``, under the model nodes' names and signatures."""

    def chat(request: dict) -> str:
        return models.chat(request)

    def detector(patch, request: dict) -> dict:
        return models.detector(patch, request)

    def embedder(text: str):
        return models.embedder(text)

    return chat, detector, embedder
