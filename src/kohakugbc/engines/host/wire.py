"""The wire between a remote unit and an engine host.

Messages are pickled tuples over ``multiprocessing.connection`` (length-framed, authenticated by
a shared key; the key is what makes unpickling safe: only holders of the key can send).

- unit: ``("hello", unit_name, {"transport"})``; host: ``("welcome", {"kind", "max_batch",
  "protocol", "spec", "shape"})`` or ``("refused", reason)``. ``shape`` is the lane's side of
  the unit's slot when the host was given it (adapter, former, select, lane), else None.
- unit: ``("submit", ticket, packet)``, ``("cancel", ticket)``, ``("start",)``, ``("stop",)``,
  ``("release", ticket)``, ``("bye",)``.
- host: ``("done", ticket, packet)``, ``("release", ticket)``; ``("state", {"capacity",
  "running", "waiting", "ready", "error", "seen", "extra"})``, where ``seen`` counts the submits
  the host has taken from this connection and ``error`` is why the unit failed to start.

A packet (``togeari.engines.common.transport.Packet``) carries the payloads of a submit or the
results of a ticket. Under the ``shm`` transport its buffers of ``SHM_THRESHOLD`` bytes and up
(numpy arrays, tensors) travel in shared-memory blocks the writer holds until the reader has
copied them out and sent ``("release", ticket)``; under ``inband`` everything is in the message.
``shm`` needs both ends on one node.

A result is ``("ok", raw)``, ``("item", message)``, ``("unit", message, retry_after)`` or
``("cancelled",)``.
"""

from togeari import Cancelled, ItemError, Ok, UnitError
from togeari.engines.common.transport import SHM_THRESHOLD, Codec

try:
    from togeari.engines.torch.codec import tensor_codec
except ImportError:
    tensor_codec = None

TRANSPORTS = ("inband", "shm")
# What a send or receive raises on a connection that is gone: closed by this process's other
# thread under a blocked call, multiprocessing.connection raises TypeError (its handle is None)
CLOSED = (OSError, EOFError, TypeError)


class RemoteError(RuntimeError):
    """An error raised on the engine host, by its message."""


def address(text: str) -> tuple[tuple[str, int], str | None]:
    """``host:port`` or ``host:port/unit`` as ``((host, port), unit)``."""
    where, _, unit = text.partition("/")
    host, _, port = where.rpartition(":")
    return (host or "127.0.0.1", int(port)), unit or None


def codec(transport: str) -> Codec:
    """The packet codec of ``transport`` (tensors as raw bytes when torch is installed)."""
    if transport not in TRANSPORTS:
        raise ValueError(f"transport {transport!r}: one of {TRANSPORTS}")
    shm = transport == "shm"
    if tensor_codec is not None:
        return tensor_codec(shm=shm, threshold=SHM_THRESHOLD)
    return Codec(shm=shm, threshold=SHM_THRESHOLD)


def encode(result) -> tuple:
    match result:
        case Ok():
            return ("ok", result.value)
        case Cancelled():
            return ("cancelled",)
        case UnitError():
            return ("unit", repr(result.error), result.retry_after)
        case ItemError():
            return ("item", repr(result.error))
    return ("item", f"unknown result {type(result).__name__}")


def decode(wire: tuple):
    match wire:
        case ("ok", raw):
            return Ok(raw)
        case ("cancelled",):
            return Cancelled()
        case ("unit", message, retry_after):
            return UnitError(RemoteError(message), retry_after=retry_after)
        case ("item", message):
            return ItemError(RemoteError(message))
    return ItemError(RemoteError(f"unreadable result {wire!r}"))
