"""The wire between a remote unit and an engine host.

Messages are pickled tuples over ``multiprocessing.connection`` (length-framed, authenticated by
a shared key; the key is what makes unpickling safe: only holders of the key can send).

- unit: ``("hello", unit_name)``; host: ``("welcome", {"kind", "max_batch", "protocol",
  "spec"})`` or ``("refused", reason)``.
- unit: ``("submit", ticket, [payload, ...])``, ``("cancel", ticket)``, ``("bye",)``.
- host: ``("done", ticket, [result, ...])``; ``("state", {"capacity", "running", "waiting",
  "ready", "seen", "extra"})``, where ``seen`` counts the submits the host has taken from this
  connection.

A result is ``("ok", raw)``, ``("item", message)``, ``("unit", message, retry_after)`` or
``("cancelled",)``.
"""

from togeari import Cancelled, ItemError, Ok, UnitError


class RemoteError(RuntimeError):
    """An error raised on the engine host, by its message."""


def address(text: str) -> tuple[tuple[str, int], str | None]:
    """``host:port`` or ``host:port/unit`` as ``((host, port), unit)``."""
    where, _, unit = text.partition("/")
    host, _, port = where.rpartition(":")
    return (host or "127.0.0.1", int(port)), unit or None


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
