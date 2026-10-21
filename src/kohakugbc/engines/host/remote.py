"""The remote unit: a lane's unit whose engine runs under an engine host on another node.

``submit`` hands the work to a sender thread (payloads are prepared there, never on the
dispatcher) and returns a ticket; a reader thread answers each ticket from the host's results.
``capacity()`` is the host's last reported capacity less the submits the host had not taken
yet. A lost connection answers every open ticket with ``UnitError``; ``ready()`` reconnects.
"""

import contextlib
import itertools
import queue
import threading
import time
from multiprocessing.connection import Client

from togeari import EngineBase, Load, UnitError

from kohakugbc.engines.host import wire


class RemoteUnit(EngineBase):
    shares_memory = False

    def __init__(
        self,
        address: str,
        authkey: bytes,
        kind: str,
        max_batch: int = 1,
        protocol: str | None = None,
        name: str | None = None,
        connect_timeout_s: float = 10.0,
        retry_s: float = 1.0,
    ):
        self.address, self.unit = wire.address(address)
        super().__init__(name or f"remote:{address}")
        self.authkey = authkey
        self.kind, self.max_batch, self.protocol = kind, max_batch, protocol
        self.connect_timeout_s, self.retry_s = connect_timeout_s, retry_s

        self.lock = threading.Lock()
        self.conn = None
        self.tickets: dict[int, tuple[int, object]] = {}
        self.counter = itertools.count(1)
        self.sent = 0
        self.state = {"capacity": 0, "running": 0, "waiting": 0, "ready": False, "seen": 0}
        self.outbox: queue.SimpleQueue = queue.SimpleQueue()
        self.host_spec: dict = {}
        self.last_try = 0.0

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        while True:
            try:
                self._connect()
                return
            except (OSError, EOFError, ConnectionError) as exc:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"{self.name}: cannot reach the host: {exc}") from exc
                time.sleep(0.2)

    def _connect(self) -> None:
        conn = Client(self.address, authkey=self.authkey)
        conn.send(("hello", self.unit))
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
        self.host_spec = info["spec"]
        with self.lock:
            self.conn = conn
            self.sent = 0
            self.state = {"capacity": 0, "running": 0, "waiting": 0, "ready": True, "seen": 0}
        self.outbox = queue.SimpleQueue()
        threading.Thread(target=self._read, args=(conn,), daemon=True).start()
        threading.Thread(target=self._send, args=(conn, self.outbox), daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            conn, self.conn = self.conn, None
        if conn is not None:
            self.outbox.put(("bye",))
            self.outbox.put(None)
        self._fail_open("stopped")

    def ready(self) -> bool:
        with self.lock:
            up = self.conn is not None and self.state["ready"]
            down = self.conn is None
        if down and time.monotonic() - self.last_try >= self.retry_s:
            self.last_try = time.monotonic()
            try:
                self._connect()
            except (OSError, EOFError, ConnectionError):
                return False
            return True
        return up

    def spec(self) -> dict:
        return {
            "engine": "remote",
            "address": f"{self.address[0]}:{self.address[1]}",
            "unit": self.unit,
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
                extra={"host": self.spec()["address"], "host_extra": s.get("extra", {})},
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
                    message = ("submit", ticket, [item.payload() for item in items])
                except Exception as exc:  # noqa: BLE001
                    self._answer(ticket, ("item", f"prepare: {exc!r}"))
                    continue
            try:
                conn.send(message)
            except (OSError, EOFError, BrokenPipeError):
                self._lost(conn)
                return

    def _read(self, conn) -> None:
        while True:
            try:
                message = conn.recv()
            except (OSError, EOFError):
                self._lost(conn)
                return
            match message[0]:
                case "done":
                    _, ticket, results = message
                    with self.lock:
                        entry = self.tickets.pop(ticket, None)
                    if entry is not None:
                        entry[1](ticket, [wire.decode(r) for r in results])
                case "state":
                    with self.lock:
                        before = self.state["capacity"] - (self.sent - self.state["seen"])
                        self.state = message[1]
                        after = self.state["capacity"] - (self.sent - self.state["seen"])
                    if after > before:
                        self.wake()

    def _answer(self, ticket, result_wire) -> None:
        with self.lock:
            entry = self.tickets.pop(ticket, None)
        if entry is not None:
            entry[1](ticket, [wire.decode(result_wire)] * entry[0])

    def _lost(self, conn) -> None:
        with self.lock:
            if self.conn is not conn:
                return
            self.conn = None
        with contextlib.suppress(OSError):
            conn.close()
        self._fail_open("connection lost")

    def _fail_open(self, why: str) -> None:
        with self.lock:
            open_, self.tickets = self.tickets, {}
        for ticket, (n, done) in open_.items():
            done(ticket, [UnitError(ConnectionError(f"{self.name}: {why}"))] * n)
