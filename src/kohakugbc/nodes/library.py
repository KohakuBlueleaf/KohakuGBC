"""The node library ``gbc``.

``build(models=None)`` gathers every node into a ``NodeLibrary``. Model nodes (``chat``,
``detector``, ``embedder``) are bound to engines on a machine; given ``models``, they call it
(the KIR reference interpreter, replay).
"""

from kohakugbc.nodes import images, regions, relations, root, rounds
from kohakugbc.nodes.registry import Models, build

MODULES = (images, regions, relations, root, rounds)

__all__ = ["MODULES", "Models", "build"]
