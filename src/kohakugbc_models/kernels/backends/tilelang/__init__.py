"""TileLang kernels (sm70 / sm75, where Triton has no tensor-core dots; registered on sm80+
too). The probe needs TileLang and the host compiler for nvcc; a kernel that fails to build is
handled by the registry's fallback."""

import shutil

import tilelang  # noqa: F401  (the backend is unavailable without it)

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends import load_tree


def probe(device: str) -> str | None:
    if not device.startswith("cuda"):
        return "TileLang targets CUDA"
    if shutil.which("nvcc") is None:
        return "nvcc not on PATH"
    return None


registry.register_probe("tilelang", probe)
FAILED = load_tree(__name__, __path__)
