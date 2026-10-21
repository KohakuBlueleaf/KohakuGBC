"""Chat protocols: a GBC request (``kohakugbc.nodes.specs.chat``) as a server's HTTP request, and
the server's reply as text.

``openai``: OpenAI-compatible ``/chat/completions``. Servers differ in where a JSON schema goes
and how Qwen3.5's thinking is turned off; the flavour picks:

| Flavour | JSON schema | Thinking off |
|---|---|---|
| ``vllm`` | ``structured_outputs: {"json": schema}`` | template switch |
| ``sglang`` | ``response_format`` (``json_schema``) | template switch |
| ``llama.cpp`` | ``json_schema`` | template switch |
| ``openai`` | ``response_format`` (``json_schema``, strict) | not sent |
| ``openrouter`` | ``response_format`` (``json_schema``) | ``reasoning: {"enabled": false}`` |

The template switch is ``chat_template_kwargs: {"enable_thinking": false}``.

``anthropic``: the Messages API (``/messages``); no schema field: the prompt asks for JSON and
the reply parsers read the first JSON object.

Every request carries ``X-GBC-Request``, the SHA-256 of its key, so a server (the replay
server, a cache, a log) can tell which question it is.
"""

import base64
import hashlib

from kohakugbc.nodes.specs import chat as specs

FLAVOURS = ("vllm", "sglang", "llama.cpp", "openai", "openrouter")


def request_id(request: dict) -> str:
    return hashlib.sha256(request["key"].encode("utf-8")).hexdigest()


def _b64(picture: dict) -> str:
    if picture is None:
        raise ValueError("an empty crop cannot be sent")
    return base64.b64encode(picture["jpeg"]).decode("ascii")


# ---- OpenAI-compatible ---------------------------------------------------------------------------


def openai_body(request: dict, model: str, flavour: str, sampling: dict) -> dict:
    content = []
    for label, picture in request["images"]:
        if label is not None:
            content.append({"type": "text", "text": label})
        url = f"data:image/jpeg;base64,{_b64(picture)}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": request["query"]})
    messages = []
    if request["system"] is not None:
        messages.append({"role": "system", "content": request["system"]})
    messages.append({"role": "user", "content": content})

    body = {
        "model": model,
        "messages": messages,
        "temperature": request["temperature"],
        "max_tokens": request["max_tokens"],
        **sampling,
    }
    schema = specs.schema(request["schema"]) if request["schema"] else None
    format_ = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    match flavour:
        case "vllm":
            if schema is not None:
                body["structured_outputs"] = {"json": schema}
            body["chat_template_kwargs"] = {"enable_thinking": False}
        case "sglang":
            if schema is not None:
                body["response_format"] = format_
            body["chat_template_kwargs"] = {"enable_thinking": False}
        case "llama.cpp":
            if schema is not None:
                body["json_schema"] = schema
            body["chat_template_kwargs"] = {"enable_thinking": False}
        case "openai":
            if schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema, "strict": True},
                }
        case "openrouter":
            if schema is not None:
                body["response_format"] = format_
            body["reasoning"] = {"enabled": False}
        case _:
            raise ValueError(f"flavour {flavour!r}: one of {FLAVOURS}")
    return body


def openai_reply(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


# ---- Anthropic Messages ------------------------------------------------------------------------


def anthropic_body(request: dict, model: str, flavour: str, sampling: dict) -> dict:
    content = []
    for label, picture in request["images"]:
        if label is not None:
            content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(picture)},
            }
        )
    content.append({"type": "text", "text": request["query"]})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": request["temperature"],
        "max_tokens": request["max_tokens"],
        **sampling,
    }
    if request["system"] is not None:
        body["system"] = request["system"]
    return body


def anthropic_reply(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")


PROTOCOLS = {
    "openai": ("/chat/completions", openai_body, openai_reply),
    "anthropic": ("/messages", anthropic_body, anthropic_reply),
}
