"""A chat server that answers from a trace: the ``api`` entry's other end, with no model.

It speaks the OpenAI-compatible ``/v1/chat/completions`` and Anthropic's ``/v1/messages``,
finds the request by its ``X-GBC-Request`` header (the SHA-256 of the request's key), waits the
recorded latency over ``speed`` (or ``latency_s``), and answers the recorded reply. It checks
what arrived against what the source sent: the query text and the decoded picture sizes; each
difference is kept in ``mismatches``. An unknown request is a 404.

    python -m kohakugbc.engines.replay.server runs/traces/poc30/text --port 8088
"""

import argparse
import base64
import hashlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from kohakugbc.nodes.specs.trace import Trace


def _pictures(body: dict) -> list[list[int]]:
    """The decoded sizes of the JPEG pictures in a request body, in order."""
    sizes = []
    for message in body.get("messages") or []:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            match part.get("type"):
                case "image_url":
                    data = part["image_url"]["url"].split(",", 1)[1]
                case "image":
                    data = part["source"]["data"]
                case _:
                    continue
            with Image.open(io.BytesIO(base64.b64decode(data))) as img:
                sizes.append([img.width, img.height])
    return sizes


def _query(body: dict) -> str | None:
    """The last text part of the last message: where GBC puts the query."""
    content = (body.get("messages") or [{}])[-1].get("content")
    if isinstance(content, list):
        texts = [p.get("text") for p in content if p.get("type") == "text"]
        return texts[-1] if texts else None
    return content


class ReplayServer:
    def __init__(self, trace: str, speed: float = 1.0, latency_s: float | None = None):
        self.trace = Trace(trace)
        self.speed, self.latency_s = speed, latency_s
        self.by_id = {
            hashlib.sha256(k.encode("utf-8")).hexdigest(): r
            for k, r in self.trace.vlm.items()
            if k is not None
        }
        self.mismatches: list[dict] = []
        self.counts = {"requests": 0, "unknown": 0, "running": 0}
        self.lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None

    def answer(self, path: str, rid: str | None, body: dict) -> tuple[int, dict]:
        with self.lock:
            self.counts["requests"] += 1
        rec = self.by_id.get(rid or "")
        if rec is None:
            with self.lock:
                self.counts["unknown"] += 1
            return 404, {"error": {"message": f"no recorded request {rid}"}}

        # What arrived against what the source sent
        sent = rec["request"].get("images")
        if sent is not None and rec.get("error") is None:
            sent = [sent] if sent and isinstance(sent[0], int) else sent
            got = _pictures(body)
            if got != sent:
                self._mismatch(rec, "images", sent, got)
        if rec["request"].get("query") not in (None, _query(body)):
            self._mismatch(rec, "query", rec["request"]["query"], _query(body))

        delay = (
            self.latency_s if self.latency_s is not None else (rec.get("latency") or 0) / self.speed
        )
        with self.lock:
            self.counts["running"] += 1
        time.sleep(delay)
        with self.lock:
            self.counts["running"] -= 1
        reply = rec["reply"] or ""
        if path.endswith("/messages"):
            return 200, {"content": [{"type": "text", "text": reply}], "stop_reason": "end_turn"}
        return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}]}

    def _mismatch(self, rec: dict, field: str, recorded, got) -> None:
        with self.lock:
            self.mismatches.append(
                {"key": rec["key"], "field": field, "recorded": recorded, "got": got}
            )

    # ---- HTTP --------------------------------------------------------------------------------

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status: int, data: dict | str, kind="application/json") -> None:
                payload = (data if isinstance(data, str) else json.dumps(data)).encode()
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                status, data = server.answer(self.path, self.headers.get("X-GBC-Request"), body)
                self._send(status, data)

            def do_GET(self):
                match self.path.rstrip("/"):
                    case "/health" | "/v1/health":
                        self._send(200, {"ok": True})
                    case "/v1/models":
                        self._send(200, {"data": [{"id": "replay"}]})
                    case "/metrics":
                        running = server.counts["running"]
                        text = f'vllm:num_requests_running{{model_name="replay"}} {running}\n'
                        text += 'vllm:num_requests_waiting{model_name="replay"} 0\n'
                        self._send(200, text, "text/plain")
                    case _:
                        self._send(404, {"error": "not found"})

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://{host}:{self.httpd.server_address[1]}/v1"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    server = ReplayServer(args.trace, args.speed)
    print("serving", server.start(args.host, args.port), flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
