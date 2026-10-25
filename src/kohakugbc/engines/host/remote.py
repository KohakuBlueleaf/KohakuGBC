"""The remote unit: a lane's unit whose engine runs under an engine host in another process.

``submit`` hands the work to a sender thread (payloads are prepared and encoded there, never on
the dispatcher) and returns a ticket; a reader thread answers each ticket from the host's
results. ``capacity()`` is the host's last reported capacity less the submits the host had not
taken yet. A lost connection answers every open ticket with ``UnitError``; ``ready()`` starts a
reconnect in the background (``locate`` first starts a spawned host's ended process again).

``manage``: the unit starts and stops the hosted unit (``start()`` asks the host and waits up to
``start_timeout_s`` for it to be up; a lazy host loads nothing before that).
"""

import contextlib
import itertools
import queue
import threading
import time
from multiprocessing.connection import Client

from togeari import EngineBase, Load, UnitError
from togeari.engines.common.transport import release

from kohakugbc.engines.host import wire


class RemoteUnit(EngineBase):
    shares_memory = False

    def __init__(
        self,
        address: str | None,
        authkey: bytes,
        kind: str,
        max_batch: int = 1,
        protocol: str | None = None,
        name: str | None = None,
        connect_timeout_s: float = 10.0,
        retry_s: float = 1.0,
        transport: str = "inband",
        unit: str | None = None,
        locate=None,
        manage: bool = False,
        start_timeout_s: float = 1800.0,
        before_start=None,
        on_start=None,
    ):
        """``address``: ``host:port/unit``; or None with ``locate``, a callable giving
        ``(host, port)`` (reviving its host when that process ended) and ``unit`` the
        hosted unit's name. A managed start first calls ``before_start(timeout_s)``, and
        afterwards ``on_start(unit, error)`` (error None once up)."""
        if address is not None:
            self.address, self.unit = wire.address(address)
        else:
            self.address, self.unit = None, unit
        super().__init__(name or f"remote:{address or unit}")
        self.authkey = authkey
        self.kind, self.max_batch, self.protocol = kind, max_batch, protocol
        self.connect_timeout_s, self.retry_s = connect_timeout_s, retry_s
        self.codec = wire.codec(transport)
        self.transport = transport
        self.locate, self.manage, self.start_timeout_s = locate, manage, start_timeout_s
        self.before_start, self.on_start = before_start, on_start

        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.conn = None
        self.tickets: dict[int, tuple[int, object]] = {}
        # Payload blocks by ticket, held until the host has read them
        self.held: dict[int, list] = {}
        self.counter = itertools.count(1)
        self.sent = 0
        self.state = self._fresh()
        self.outbox: queue.SimpleQueue = queue.SimpleQueue()
        self.host_spec: dict = {}
        self.shape = None
        self.last_try = 0.0
        self.reconnecting = False
        self.wanted = False

    @staticmethod
    def _fresh() -> dict:
        return {"capacity": 0, "running": 0, "waiting": 0, "ready": False, "seen": 0}

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        while True:
            try:
                self._connect()
                break
            except (OSError, EOFError, ConnectionError) as exc:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"{self.name}: cannot reach the host: {exc}") from exc
                time.sleep(0.2)
        if self.manage:
            self.wanted = True
            try:
                if self.before_start is not None:
                    self.before_start(self.start_timeout_s)
                self.outbox.put(("start",))
                self._wait_up()
            except Exception as exc:
                if self.on_start is not None:
                    self.on_start(self.unit, f"{type(exc).__name__}: {exc}")
                raise
            if self.on_start is not None:
                self.on_start(self.unit, None)

    def _wait_up(self) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        with self.changed:
            while not self.state["ready"]:
                if self.state.get("error"):
                    raise RuntimeError(f"{self.name}: {self.state['error']}")
                if self.conn is None:
                    raise ConnectionError(f"{self.name}: host lost while starting")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"{self.name}: not up in {self.start_timeout_s}s")
                self.changed.wait(min(left, 1.0))

    def describe(self) -> dict:
        """Connect once and return the welcome (kind, batch size, protocol, spec, shape)."""
        conn = self._open()
        try:
            return conn.recv()[1]
        finally:
            with contextlib.suppress(OSError, EOFError):
                conn.send(("bye",))
                conn.close()

    def _where(self) -> tuple[str, int]:
        return self.locate() if self.locate is not None else self.address

    def _open(self):
        conn = Client(self._where(), authkey=self.authkey)
        conn.send(("hello", self.unit, {"transport": self.transport}))
        return conn

    def _connect(self) -> None:
        conn = self._open()
        reply = conn.recv()
        if reply[0] != "welcome":
            conn.close()
            raise ConnectionError(f"{self.name}: host refused: {reply[1]}")
        info = reply[1]
        if info["kind"] != self.kind:
            conn.close()
            raise ConnectionError(
                f"{self.name}: host unit is {info['kind']}, lane wants {self.kind}"
            )
        self.host_spec, self.shape = info["spec"], info.get("shape")
        with self.lock:
            self.conn = conn
            self.sent = 0
            self.state = self._fresh()
        self.outbox = queue.SimpleQueue()
        threading.Thread(target=self._read, args=(conn,), daemon=True).start()
        threading.Thread(target=self._send, args=(conn, self.outbox), daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            conn, self.conn = self.conn, None
        if conn is not None:
            if self.manage:
                self.outbox.put(("stop",))
            self.outbox.put(("bye",))
            self.outbox.put(None)
        self.wanted = False
        self._fail_open("stopped")

    def ready(self) -> bool:
        """Up and connected; when the connection is down, one reconnect starts on a thread of
        its own (no sooner than ``retry_s`` after the last), so a host started again never
        holds the caller."""
        with self.lock:
            up = self.conn is not None and self.state["ready"]
            due = self.conn is None and not self.reconnecting
            due = due and time.monotonic() - self.last_try >= self.retry_s
            if due:
                self.reconnecting, self.last_try = True, time.monotonic()
        if due:
            threading.Thread(
                target=self._reconnect, name=f"{self.name}-reconnect", daemon=True
            ).start()
        return up

    def _reconnect(self) -> None:
        try:
            self._connect()
            if self.manage and self.wanted:
                self.outbox.put(("start",))
        except (OSError, EOFError, ConnectionError, RuntimeError, TimeoutError):
            pass
        finally:
            with self.lock:
                self.reconnecting = False

    def spec(self) -> dict:
        where = self._where() if self.locate is None else ("spawned", self.unit)
        return {
            "engine": "remote",
            "address": f"{where[0]}:{where[1]}",
            "unit": self.unit,
            "transport": self.transport,
            "hosted": self.host_spec,
        }

    # ---- the contract ------------------------------------------------------------------------

    def capacity(self) -> int:
        with self.lock:
            if self.conn is None:
                return 0
            return max(0, self.state["capacity"] - (self.sent - self.state["seen"]))

    def load(self) -> Load:
        with self.lock:
            s = self.state
            return Load(
                running=s["running"],
                waiting=s["waiting"] + (self.sent - s["seen"]),
                extra=dict(s.get("extra") or {}),
            )

    def submit(self, work, done):
        items = work if isinstance(work, list) else [work]
        ticket = next(self.counter)
        with self.lock:
            if self.conn is None:
                down = True
            else:
                down = False
                self.tickets[ticket] = (len(items), done)
                self.sent += 1
        if down:
            done(ticket, [UnitError(ConnectionError(f"{self.name} is not connected"))] * len(items))
            return ticket
        self.outbox.put(("submit", ticket, items))
        return ticket

    def cancel(self, ticket) -> None:
        with self.lock:
            open_ = ticket in self.tickets and self.conn is not None
        if open_:
            self.outbox.put(("cancel", ticket))

    # ---- threads -----------------------------------------------------------------------------

    def _send(self, conn, outbox: queue.SimpleQueue) -> None:
        while (message := outbox.get()) is not None:
            if message[0] == "submit":
                _, ticket, items = message
                try:
                    packet, owned = self.codec.encode([item.payload() for item in items])
                except Exception as exc:  # noqa: BLE001
                    self._answer(ticket, ("item", f"prepare: {exc!r}"))
                    continue
                if owned:
                    with self.lock:
                        self.held[ticket] = owned
                message = ("submit", ticket, packet)
            try:
                conn.send(message)
            except wire.CLOSED:
                self._lost(conn)
                return

    def _read(self, conn) -> None:
        while True:
            try:
                message = conn.recv()
            except wire.CLOSED:
                self._lost(conn)
                return
            match message[0]:
                case "done":
                    _, ticket, packet = message
                    try:
                        results, _ = self.codec.decode(packet, copy=True)
                    except Exception as exc:  # noqa: BLE001
                        results = None
                        failure = ("item", f"decode: {exc!r}")
                    if packet.blocks:
                        self.outbox.put(("release", ticket))
                    with self.lock:
                        entry = self.tickets.pop(ticket, None)
                    if entry is not None:
                        out = results if results is not None else [failure] * entry[0]
                        entry[1](ticket, [wire.decode(r) for r in out])
                case "release":
                    with self.lock:
                        owned = self.held.pop(message[1], [])
                    release(owned, unlink=True)
                case "state":
                    with self.changed:
                        before = self.state["capacity"] - (self.sent - self.state["seen"])
                        self.state = message[1]
                        after = self.state["capacity"] - (self.sent - self.state["seen"])
                        self.changed.notify_all()
                    if after > before:
                        self.wake()

    def _answer(self, ticket, result_wire) -> None:
        with self.lock:
            entry = self.tickets.pop(ticket, None)
        if entry is not None:
            entry[1](ticket, [wire.decode(result_wire)] * entry[0])

    def _lost(self, conn) -> None:
        with self.changed:
            if self.conn is not conn:
                return
            self.conn = None
            self.changed.notify_all()
        with contextlib.suppress(OSError):
            conn.close()
        self._fail_open("connection lost")

    def _fail_open(self, why: str) -> None:
        with self.lock:
            open_, self.tickets = self.tickets, {}
            held, self.held = list(self.held.values()), {}
        for owned in held:
            release(owned, unlink=True)
        for ticket, (n, done) in open_.items():
            done(ticket, [UnitError(ConnectionError(f"{self.name}: {why}"))] * n)
