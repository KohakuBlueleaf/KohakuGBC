"""The ``api`` entry of the ``vlm`` slot: Togeari's web-call engine per endpoint, speaking a
chat protocol.

Settings: ``protocol`` (``"openai"`` or ``"anthropic"``), ``flavour`` (OpenAI-compatible
servers: ``vllm``, ``sglang``, ``llama.cpp``, ``openai``, ``openrouter``), ``model``,
``sampling`` (extra body fields), ``structured`` (``grammar``: the request's JSON schema goes to
the server; ``prompt``: the prompt alone asks for the JSON), ``endpoints``: a list of
``{"base_url", "max_inflight" (32), "rate" (requests per second, or ``[rate, burst]``),
"api_key_env", "headers"}``; ``timeout_s`` (600), ``connect_timeout_s`` (10), ``health_path``,
``metrics`` (``"vllm"``: the server's running and waiting counts in the unit's load report).
"""

import os
import re

import httpx
from togeari.adapters.http import HttpRequest
from togeari.engines.web import WebCallEngine

from kohakugbc.engines.registry import Slot, entry
from kohakugbc.engines.vlm import protocols


class ChatAdapter:
    """The ``chat`` node's request as an HTTP request of the protocol; the reply as text."""

    protocols = {"http"}
    light = True

    def __init__(
        self,
        protocol: str,
        model: str,
        flavour: str = "vllm",
        sampling=None,
        structured: str = "grammar",
    ):
        if structured not in STRUCTURED:
            raise ValueError(f"structured {structured!r}: one of {STRUCTURED}")
        self.path, self.build, self.read = protocols.PROTOCOLS[protocol]
        self.model, self.flavour = model, flavour
        self.sampling = dict(sampling or {})
        self.grammar = structured == "grammar"

    def check(self, signature) -> list[str]:
        return []

    def prepare(self, inputs: dict, protocol: str) -> HttpRequest:
        request = inputs["request"]
        body = self.build(request, self.model, self.flavour, self.sampling, self.grammar)
        return HttpRequest(
            "POST", self.path, json=body, headers={"X-GBC-Request": protocols.request_id(request)}
        )

    def decode(self, raw, protocol: str) -> str:
        return self.read(raw.json())


STRUCTURED = ("grammar", "prompt")
METRIC = re.compile(r"^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.eE+-]+)", re.M)


def vllm_metrics(base_url: str):
    """A load probe reading vLLM's Prometheus counters of running and waiting requests."""
    root = base_url.rstrip("/").removesuffix("/v1")

    def probe() -> dict:
        text = httpx.get(root + "/metrics", timeout=2.0).text
        out = {"running": 0.0, "waiting": 0.0}
        for name, value in METRIC.findall(text):
            out[name] += float(value)
        return out

    return probe


def _headers(endpoint: dict, protocol: str) -> dict:
    headers = dict(endpoint.get("headers") or {})
    env = endpoint.get("api_key_env")
    if env:
        key = os.environ[env]
        if protocol == "anthropic":
            headers.setdefault("x-api-key", key)
            headers.setdefault("anthropic-version", "2023-06-01")
        else:
            headers.setdefault("Authorization", f"Bearer {key}")
    return headers


@entry("vlm", "api")
def api(settings: dict, shared: dict) -> Slot:
    protocol = settings.get("protocol", "openai")
    if protocol not in protocols.PROTOCOLS:
        raise ValueError(f"protocol {protocol!r}: one of {sorted(protocols.PROTOCOLS)}")
    units = []
    for i, endpoint in enumerate(settings["endpoints"]):
        rate = endpoint.get("rate")
        probe = None
        if settings.get("metrics") == "vllm":
            probe = vllm_metrics(endpoint["base_url"])
        units.append(
            WebCallEngine(
                endpoint["base_url"].rstrip("/"),
                max_inflight=int(endpoint.get("max_inflight", 32)),
                rate=tuple(rate) if isinstance(rate, list) else rate,
                headers=_headers(endpoint, protocol),
                timeout_s=float(settings.get("timeout_s", 600)),
                connect_timeout_s=float(settings.get("connect_timeout_s", 10)),
                health_path=settings.get("health_path"),
                load_probe=probe,
                name=endpoint.get("name", f"api-{i}"),
            )
        )
    adapter = ChatAdapter(
        protocol,
        settings["model"],
        settings.get("flavour", "vllm"),
        settings.get("sampling"),
        settings.get("structured", "grammar"),
    )
    return Slot(units=units, adapter=adapter, lane=dict(settings.get("lane") or {}))
