"""Op registry: backends register implementations; a caller resolves an op once to a ``Bound``
that calls its implementation and, when that backend cannot build or launch, records it and
rebinds (with every live ``Bound`` of the same query on that backend) to the next backend of
the arch's priority.

- ``register(op, backend, fn, supports)``: ``supports(query)`` is None when the backend serves
  the query, else the reason it does not.
- ``resolve(op, device, dtype, backend=None, **static)``: the first backend of the priority that
  is loaded, supports the query, passes its once-per-device probe and has not failed it here; a
  pinned ``backend`` gets no fallback.
- ``available(...)``: every backend's reason (None: usable).

A failure is in-process: a build / launch error (``errors.is_backend_failure``) rebinds and is
remembered for the query's (op, arch, dtype, static options); an out-of-memory or a sticky
device error propagates and disables nothing.
"""

import importlib
import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch

from kohakugbc_models.kernels.errors import BackendUnavailable, is_backend_failure
from kohakugbc_models.kernels.hardware import arch_of, family_of

BACKENDS = ("gluon", "tilelang", "triton", "compile", "eager")
PRIORITY: dict[str, tuple[str, ...]] = {
    "sm70": ("tilelang", "compile", "eager"),
    "sm75": ("tilelang", "compile", "eager"),
    "sm80": ("triton", "tilelang", "compile", "eager"),
    "sm86": ("triton", "tilelang", "compile", "eager"),
    "sm89": ("triton", "tilelang", "compile", "eager"),
    "sm90": ("triton", "tilelang", "compile", "eager"),
    "sm10x": ("gluon", "triton", "tilelang", "compile", "eager"),
    "sm12x": ("triton", "tilelang", "compile", "eager"),
    "xpu": ("triton", "compile", "eager"),
    "cpu": ("eager",),
}
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Query:
    op: str
    arch: str
    dtype: torch.dtype
    static: tuple = ()

    def get(self, key: str, default=None):
        return dict(self.static).get(key, default)


@dataclass
class Impl:
    backend: str
    fn: Callable
    supports: Callable[[Query], str | None] | None = None


IMPLS: dict[str, dict[str, Impl]] = {}
UNAVAILABLE: dict[str, str] = {}  # backend -> why it could not be loaded
FAILED: dict[tuple, str] = {}  # (query, backend) -> the failure met in this process
PROBES: dict[str, Callable[[str], str | None]] = {}  # backend -> probe(device)
_PROBED: dict[tuple[str, str], str | None] = {}
_LIVE: "weakref.WeakSet[Bound]" = weakref.WeakSet()
_LOAD = threading.Lock()
_LOADED: list[bool] = []


class NoImplementation(RuntimeError):
    """No backend can serve the query; the message lists each backend's reason."""


def register(op: str, backend: str, fn: Callable, supports=None) -> None:
    IMPLS.setdefault(op, {})[backend] = Impl(backend, fn, supports)


def register_probe(backend: str, probe: Callable[[str], str | None]) -> None:
    PROBES[backend] = probe


def load_backends() -> None:
    """Import every backend package once per process; a failed import marks the backend
    unavailable (the rest still register)."""
    if _LOADED:
        return
    with _LOAD:
        if _LOADED:
            return
        for name in BACKENDS:
            try:
                importlib.import_module(f"kohakugbc_models.kernels.backends.{name}")
            except ImportError as err:
                UNAVAILABLE[name] = f"{type(err).__name__}: {err}"
        _LOADED.append(True)


def probe(backend: str, device) -> str | None:
    key = (backend, str(torch.device(device)))
    if key not in _PROBED:
        check = PROBES.get(backend)
        _PROBED[key] = None if check is None else check(str(torch.device(device)))
        if _PROBED[key]:
            log.warning("backend %s cannot run on %s: %s", backend, key[1], _PROBED[key])
    return _PROBED[key]


def reason(backend: str, query: Query, device) -> str | None:
    if backend in UNAVAILABLE:
        return f"unavailable ({UNAVAILABLE[backend]})"
    impl = IMPLS.get(query.op, {}).get(backend)
    if impl is None:
        return "not implemented"
    if impl.supports is not None and (why := impl.supports(query)) is not None:
        return why
    if (query, backend) in FAILED:
        return f"failed here ({FAILED[(query, backend)][:300]})"
    why = probe(backend, device)
    return None if why is None else f"cannot run on {device} ({why[:300]})"


def available(op: str, device, dtype: torch.dtype, **static) -> dict:
    load_backends()
    query = Query(op, arch_of(device), dtype, tuple(sorted(static.items())))
    return {b: reason(b, query, device) for b in BACKENDS}


def first(query: Query, device, order: tuple[str, ...]) -> str:
    reasons = []
    for name in order:
        why = reason(name, query, device)
        if why is None:
            return name
        reasons.append(f"{name}: {why}")
    raise NoImplementation(f"{query.op} on {query.arch}/{query.dtype}: " + "; ".join(reasons))


class Bound:
    """A resolved op: calls its implementation; a backend failure rebinds it."""

    def __init__(self, query: Query, device, order: tuple[str, ...], backend: str):
        self.query, self.device, self.order = query, device, order
        self.switch(backend)
        _LIVE.add(self)

    def switch(self, backend: str) -> None:
        self.backend = backend
        self.fn = IMPLS[self.query.op][backend].fn

    def __call__(self, *args, **kwargs):
        while True:
            try:
                return self.fn(*args, **kwargs)
            except BaseException as err:
                if not is_backend_failure(err):
                    raise
                self.fall_back(err)

    def fall_back(self, err: BaseException) -> None:
        """Record the failure and rebind this op and its live siblings to the next backend;
        re-raise when none is left or the backend was pinned."""
        failed = self.backend
        FAILED[(self.query, failed)] = f"{type(err).__name__}: {err}"
        rest = self.order[self.order.index(failed) + 1 :]
        try:
            name = first(self.query, self.device, rest)
        except NoImplementation as none_left:
            raise BackendUnavailable(str(err)) from none_left
        siblings = [b for b in list(_LIVE) if b.query == self.query and b.backend == failed]
        for op in siblings:
            op.switch(name)
        log.warning(
            "%s on %s/%s: backend %s failed, using %s (%d bound): %s",
            self.query.op,
            self.query.arch,
            self.query.dtype,
            failed,
            name,
            len(siblings),
            str(err)[:2000],
        )


def priority(arch: str) -> tuple[str, ...]:
    """The backend order of an arch: its own entry, else its family's, else compile and eager."""
    return PRIORITY.get(arch) or PRIORITY.get(family_of(arch), ("compile", "eager"))


def resolve(op: str, device, dtype: torch.dtype, backend: str | None = None, **static) -> Bound:
    load_backends()
    query = Query(op, arch_of(device), dtype, tuple(sorted(static.items())))
    order = (backend,) if backend else priority(query.arch)
    return Bound(query, device, order, first(query, device, order))
