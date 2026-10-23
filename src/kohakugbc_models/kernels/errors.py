"""Errors of the kernel layer: what counts as "this backend cannot run here"."""

import torch

try:
    from triton.compiler.errors import CompilationError as TritonCompileError
    from triton.runtime.errors import OutOfResources as TritonOutOfResources
except ImportError:  # no Triton in this environment
    TritonCompileError = TritonOutOfResources = RuntimeError


class BackendUnavailable(RuntimeError):
    """A backend cannot serve this op here (build, launch or probe failure)."""


class ConfigsFailed(BackendUnavailable):
    """Every shortlisted config of one tuned problem failed to build or run."""


# A build / launch failure: never an out-of-memory (that propagates and disables nothing)
BUILD_ERRORS = (RuntimeError, OSError, UnicodeError, TritonCompileError, TritonOutOfResources)
STICKY = (
    "illegal memory access",
    "misaligned address",
    "illegal instruction",
    "device-side assert",
    "unspecified launch failure",
)


def is_sticky(err: BaseException) -> bool:
    """An error after which the device context is unusable: no backend may be tried next."""
    text = str(err).lower()
    return any(s in text for s in STICKY)


def is_backend_failure(err: BaseException) -> bool:
    return (
        isinstance(err, BUILD_ERRORS)
        and not isinstance(err, torch.OutOfMemoryError)
        and not is_sticky(err)
    )
