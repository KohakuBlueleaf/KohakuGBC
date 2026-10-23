"""A GBC trace's VLM requests replayed against OpenAI-compatible servers (vLLM, SGLang, ...): the
servers alone, at a fixed number of requests in flight, spread over the least-loaded URL.

Each recorded request is rebuilt: its system prompt, query, schema, max_tokens and sampling; its
picture is the job's image cropped to the key's box (the boxes' union, or the whole image) and
resized to the recorded size, so a request carries the recorded number of image tokens.
``--repeat`` sends the trace that many times; every copy's pictures differ in one pixel, so no
copy hits another's prefix or multimodal cache (the shared system prompts still may).

Writes ``OUT.jsonl`` (one row per request: latency, ttft, tokens, text) and ``OUT.summary.json``.

    python scripts/bench/vlm/replay.py TRACE_DIR IMAGE_ROOT --url http://127.0.0.1:8000/v1 \
        --model qwen3.5-4b --flavour vllm --inflight 64 --repeat 4 --out runs/replay/vllm_c64
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

from kohakugbc.nodes.specs import chat


def crop_box(key: dict) -> list[float]:
    if "box" in key:
        return key["box"]
    if "boxes" in key:
        bs = key["boxes"]
        lo = [min(b[0] for b in bs), min(b[1] for b in bs)]
        return lo + [max(b[2] for b in bs), max(b[3] for b in bs)]
    return [0.0, 0.0, 1.0, 1.0]


def picture(image: Image.Image, key: dict, size, copy: int) -> str:
    w, h = image.size
    x0, y0, x1, y1 = crop_box(key)
    left, top = int(x0 * w), int(y0 * h)
    box = (left, top, max(int(x1 * w), left + 1), max(int(y1 * h), top + 1))
    pic = image.crop(box).resize(tuple(size), Image.BICUBIC)
    if copy:
        r, g, b = pic.getpixel((0, 0))
        pic.putpixel((0, 0), ((r + copy * 37) % 256, g, b))
    buf = io.BytesIO()
    pic.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def body(row: dict, pic: str | None, args) -> dict:
    req = row["request"]
    content = []
    if pic is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pic}"}})
    content.append({"type": "text", "text": req["query"]})
    messages = []
    if req["system"] is not None:
        messages.append({"role": "system", "content": req["system"]})
    messages.append({"role": "user", "content": content})
    out = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens or req["max_tokens"],
        **req["sampling"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if req["schema"] and not args.no_schema:
        schema = chat.schema(req["schema"])
        if args.flavour == "vllm":
            out["structured_outputs"] = {"json": schema}
        else:
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema},
            }
    return out


def valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


async def send(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    start = time.perf_counter()
    ttft, usage, text, err, finish = None, None, [], None, None
    try:
        async with client.stream("POST", url + "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                err = f"{resp.status_code} {(await resp.aread())[:300]!r}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for c in chunk.get("choices") or []:
                        finish = c.get("finish_reason") or finish
                        piece = (c.get("delta") or {}).get("content")
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            text.append(piece)
    except Exception as exc:  # noqa: BLE001 - a failed request is a row, not a crash
        err = repr(exc)
    reply = "".join(text)
    return {
        "start": start,
        "latency": time.perf_counter() - start,
        "ttft": ttft,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "valid_json": valid_json(reply),
        "finish": finish,
        "text": reply,
        "error": err,
    }


def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None


def summarize(name: str, inflight: int, results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["error"] is None]
    ptok = sum(r["prompt_tokens"] or 0 for r in ok)
    ctok = sum(r["completion_tokens"] or 0 for r in ok)
    lat = [r["latency"] for r in ok]
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    decode = [
        (r["completion_tokens"] - 1) / (r["latency"] - r["ttft"])
        for r in ok
        if r["ttft"] and (r["completion_tokens"] or 0) > 1 and r["latency"] > r["ttft"]
    ]
    return {
        "name": name,
        "inflight": inflight,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "invalid_json": sum(not r["valid_json"] for r in ok),
        "wall_s": round(wall, 2),
        "req_s": round(len(ok) / wall, 2),
        "prompt_tok_s": round(ptok / wall),
        "gen_tok_s": round(ctok / wall),
        "mean_prompt": round(ptok / max(1, len(ok)), 1),
        "mean_gen": round(ctok / max(1, len(ok)), 1),
        "lat_p50": pct(lat, 0.5),
        "lat_p90": pct(lat, 0.9),
        "ttft_p50": pct(ttft, 0.5),
        "ttft_p90": pct(ttft, 0.9),
        "decode_tok_s_per_req_p50": pct(decode, 0.5),
        "first_error": next((r["error"] for r in results if r["error"]), None),
    }


async def replay(args) -> dict:
    root = Path(args.images)
    with open(Path(args.trace) / "vlm.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    images: dict[str, Image.Image] = {}
    payloads = []
    for copy in range(args.repeat):
        for row in rows:
            key = json.loads(row["key"])
            size = row["request"]["images"]
            pic = None
            if size and not args.no_image:
                name = key["image"]
                if name not in images:
                    images[name] = Image.open(root / name).convert("RGB")
                pic = picture(images[name], key, size, copy)
            payloads.append((row["action"], body(row, pic, args)))

    urls = args.url.split(",")
    load = dict.fromkeys(urls, 0)
    gate = asyncio.Semaphore(args.inflight)
    limits = httpx.Limits(
        max_connections=args.inflight + 8, max_keepalive_connections=args.inflight + 8
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30), limits=limits) as client:

        async def one(action: str, payload: dict) -> dict:
            async with gate:
                url = min(urls, key=load.__getitem__)
                load[url] += 1
                try:
                    row = await send(client, url, payload)
                finally:
                    load[url] -= 1
                row["action"] = action
                return row

        # Warm-up requests outside the measured window
        await asyncio.gather(*(one(a, p) for a, p in payloads[:8]))
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(a, p) for a, p in payloads))
        wall = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({**r, "start": r["start"] - t0}) + "\n" for r in results)
    summary = summarize(out.name, args.inflight, results, wall)
    Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("images")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1", help="comma-separated")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--flavour", default="vllm")
    p.add_argument("--inflight", type=int, default=64)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--no-schema", action="store_true", help="no structured output (a probe)")
    p.add_argument("--no-image", action="store_true", help="no picture in any request (a probe)")
    p.add_argument("--max-tokens", type=int, default=0, help="every request's (1: prefill only)")
    args = p.parse_args()
    print(json.dumps(asyncio.run(replay(args))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
