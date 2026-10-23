"""The eager references: what each op computes, in plain torch."""

from kohakugbc_models.kernels.backends import load_tree

FAILED = load_tree(__name__, __path__)
