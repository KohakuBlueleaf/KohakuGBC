"""The node library ``gbc``.

``build(models=None)`` gathers every node into a ``NodeLibrary``. Model nodes (``chat``,
``detector``, ``embedder``, ``tagger``) are bound to engines on a machine; given ``models``, they
call it (the KIR reference interpreter, replay).
"""

from kohakugbc.nodes import images, nest, regions, relations, root, rounds, shape, tags
from kohakugbc.nodes.registry import Models, build

MODULES = (images, regions, relations, root, rounds, tags, nest, shape)

__all__ = ["MODULES", "Models", "build"]
