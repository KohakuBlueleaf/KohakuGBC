"""Plain captions (title, brief, description as one JSON object) of danbooru images on one card,
vLLM's ``AsyncLLM`` in-process with every request in flight at once: the throughput and reply
lengths a flat caption costs next to GBC, on the engine settings of the GBC machine's VLM.

The prompts are KohakuCaption's JSON captioning prompts. ``--tags none``: the image alone;
``--tags danbooru``: the post's character and copyright tags for naming and its general tags (most
popular first, ``--max-general`` of them, 0: all) as appearance hints. Pictures are resized within
``--max-short`` / ``--max-long`` before the clock; ``--frontend``: vLLM's input processing or
KohakuvLLM's (``kohakuvllm.front``). Every reply runs under KohakuvLLM's JSON grammar.

``META``: rows of id and tags by type, images ``IMAGES/<id>.webp``. Writes ``OUT.jsonl`` (per
image) and ``OUT.summary.json`` (images/s, tokens/s, token / word / character spreads, parse rate).

    python scripts/bench/caption/plain.py META IMAGES OUT --model MODEL --tags danbooru
"""

import argparse
import asyncio
import io
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from kohakugbc.nodes.specs.parse import load_json

try:
    from kohakuvllm.front import engine as kohaku_front
except ImportError:
    kohaku_front = None

SYSTEM = (
    "You write vivid, natural descriptions of visual content. "
    "Always respond in valid JSON. "
    "Never reference 'this image' or 'the image' -- just describe the content directly."
)
PLAIN = (
    "Describe what you can directly see. Do not guess names or identities.\n"
    "Respond ONLY in JSON:\n"
    '{"title": "1 sentence title", "brief": "1-2 sentence brief", '
    '"description": "3-6 sentence detailed description"}'
)
WITH_TAGS = (
    "Characters: {char_tags}\n"
    "Appearance hints (for accuracy only, do not repeat): {hint_tags}\n\n"
    "Describe the scene naturally, naming any identified characters. "
    "3-5 sentences.\n"
    "Respond ONLY in JSON:\n"
    '{{"title": "3-5 word title", "brief": "1 sentence", '
    '"description": "detailed 3-5 sentences"}}'
)
FIELDS = ("title", "brief", "description")
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in FIELDS},
    "required": list(FIELDS),
}
KOHAKU_GRAMMAR = "kohakuvllm.grammar:JsonSchemaLogits"
# Refusal phrasing, counted (not judged) per reply
REFUSAL = re.compile(
    r"I(?:'m| am) (?:sorry|unable)|I can(?:not|'t) (?:help|assist|provide|describe|comply)"
    r"|not able to (?:help|assist|describe)|(?:inappropriate|explicit|sexual) content"
    r"|against (?:my|the) (?:guidelines|policy)",
    re.IGNORECASE,
)
# A reply this short is not a caption (the three fields hold a few characters at most)
TINY_TOKENS = 40
# The GBC machine's VLM engine settings (configs/machine/qwen35_card.py)
ENGINE = {
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "mamba_ssm_cache_dtype": "bfloat16",
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": True,
    "limit_mm_per_prompt": {"image": 1, "video": 0},
    "compilation_config": {"compile_mm_encoder": True, "cudagraph_mm_encoder": True},
}


def human(tags: list[str]) -> list[str]:
    return [t.replace("_", " ") for t in tags]


def instruction(tags: dict, mode: str, max_general: int) -> str:
    if mode == "none":
        return PLAIN
    names = human(tags.get("character", []))
    if tags.get("copyright"):
        names.append("Series: " + ", ".join(human(tags["copyright"])))
    general = tags.get("general", [])
    general = general[:max_general] if max_general else general
    return WITH_TAGS.format(
        char_tags=", ".join(names) or "unknown", hint_tags=", ".join(human(general))
    )


def picture(path: Path, max_short: int, max_long: int, jpeg: bool):
    with Image.open(path) as img:
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_short / min(w, h), max_long / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    if not jpeg:
        return img
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def spread(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {
        "mean": round(statistics.mean(s), 1),
        "p10": s[len(s) // 10],
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, len(s) * 9 // 10)],
        "max": s[-1],
    }


async def run_all(args, rows, pics) -> tuple[list, dict]:
    engine = {
        **ENGINE,
        "gpu_memory_utilization": args.util,
        "max_num_seqs": args.seqs,
        "logits_processors": [KOHAKU_GRAMMAR],
        **json.loads(args.engine),
    }
    if args.frontend == "kohakuvllm":
        engine = kohaku_front.engine_args(engine)
    t0 = time.time()
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model, **engine))
    processor = AutoProcessor.from_pretrained(args.model)
    front = None
    if args.frontend == "kohakuvllm":
        front = kohaku_front.Frontend(llm, args.model, workers=args.workers)
    load_s = time.time() - t0
    params = SamplingParams(
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        max_tokens=args.max_tokens,
        extra_args={"kohaku_json_schema": json.dumps(SCHEMA)},
        logprobs=args.logprobs or None,
    )

    async def one(i: int, r: dict, pic):
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction(r["tags"], args.tags, args.max_general)},
                ],
            },
        ]
        if front is not None:
            prompt = await front.inputs(messages, [pic])
        else:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt = {"prompt": text + args.prefill, "multi_modal_data": {"image": pic}}
        final = None
        async for out in llm.generate(prompt, params, f"{args.tags}-{i}"):
            final = out
        return final

    # A few requests warm the engine, then every request is in flight at once
    await asyncio.gather(*(one(-1 - i, rows[i], pics[i]) for i in range(8)))
    t1 = time.time()
    outs = await asyncio.gather(*(one(i, r, p) for i, (r, p) in enumerate(zip(rows, pics))))
    run_s = time.time() - t1
    if front is not None:
        front.close()
    llm.shutdown()
    return outs, {"load_s": round(load_s, 1), "run_s": run_s, "engine": engine}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("meta")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--model", required=True)
    p.add_argument("--tags", choices=("none", "danbooru"), default="none")
    p.add_argument("--max-general", type=int, default=0, help="general tags given (0: all)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-short", type=int, default=512)
    p.add_argument("--max-long", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--frontend", choices=("vllm", "kohakuvllm"), default="vllm")
    p.add_argument(
        "--prefill",
        default="",
        type=lambda s: s.encode().decode("unicode_escape"),
        help="the reply's start, after the assistant header (escapes read: '```json\\n')",
    )
    p.add_argument("--workers", type=int, default=4, help="kohakuvllm frontend's processes")
    p.add_argument("--token-ids", action="store_true", help="keep each reply's sampled ids")
    p.add_argument("--logprobs", type=int, default=0, help="keep each step's top N (0: none)")
    p.add_argument("--util", type=float, default=0.90)
    p.add_argument("--seqs", type=int, default=256)
    p.add_argument("--engine", default="{}", help="more AsyncEngineArgs fields, a JSON object")
    args = p.parse_args()
    if args.frontend == "kohakuvllm" and kohaku_front is None:
        p.error("--frontend kohakuvllm needs the kohakuvllm package")
    if args.frontend == "kohakuvllm" and args.prefill:
        p.error("--prefill needs --frontend vllm (KohakuvLLM's frontend templates the messages)")

    with open(args.meta, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = rows[: args.limit] if args.limit else rows

    # Pictures decoded and resized before the clock (JPEG bytes for KohakuvLLM's frontend)
    t0 = time.time()
    jpeg = args.frontend == "kohakuvllm"
    with ThreadPoolExecutor(16) as pool:
        pics = list(
            pool.map(
                lambda r: picture(
                    Path(args.images) / f"{r['id']}.webp", args.max_short, args.max_long, jpeg
                ),
                rows,
            )
        )
    prep_s = time.time() - t0

    outs, timing = asyncio.run(run_all(args, rows, pics))
    run_s = timing["run_s"]

    records, ok, cut, refusals, tiny = [], 0, 0, 0, 0
    # Per post rating, when META carries one
    by_rating: dict = {}
    tokens = {"prompt": [], "output": []}
    words = {k: [] for k in FIELDS}
    chars = {k: [] for k in FIELDS}
    for r, out in zip(rows, outs):
        reply = out.outputs[0].text
        parsed = load_json(reply)
        good = isinstance(parsed, dict) and all(isinstance(parsed.get(k), str) for k in FIELDS)
        ok += good
        cut += out.outputs[0].finish_reason == "length"
        refused = REFUSAL.search(reply) is not None
        refusals += refused
        n_in, n_out = len(out.prompt_token_ids), len(out.outputs[0].token_ids)
        tiny += n_out < TINY_TOKENS
        if "rating" in r:
            b = by_rating.setdefault(
                str(r["rating"]), {"images": 0, "parsed": 0, "refusal_hits": 0, "tiny_replies": 0}
            )
            b["images"] += 1
            b["parsed"] += good
            b["refusal_hits"] += refused
            b["tiny_replies"] += n_out < TINY_TOKENS
        tokens["prompt"].append(n_in)
        tokens["output"].append(n_out)
        if good:
            for k in FIELDS:
                words[k].append(len(parsed[k].split()))
                chars[k].append(len(parsed[k]))
        records.append(
            {
                "id": r["id"],
                "reply": reply,
                "caption": {k: parsed.get(k) for k in FIELDS} if good else None,
                "prompt_tokens": n_in,
                "output_tokens": n_out,
                **({"token_ids": list(out.outputs[0].token_ids)} if args.token_ids else {}),
                **(
                    {
                        "top": [
                            [[int(t), float(lp.logprob)] for t, lp in step.items()]
                            for step in out.outputs[0].logprobs
                        ]
                    }
                    if args.logprobs
                    else {}
                ),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_path}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(records)
    summary = {
        "model": args.model,
        "tags": args.tags,
        "max_general": args.max_general,
        "frontend": args.frontend,
        "prefill": args.prefill,
        "images": n,
        "prep_s": round(prep_s, 1),
        "load_s": timing["load_s"],
        "run_s": round(run_s, 1),
        "images_per_s": round(n / run_s, 3),
        "output_tokens_per_s": round(sum(tokens["output"]) / run_s, 1),
        "prompt_tokens_per_s": round(sum(tokens["prompt"]) / run_s, 1),
        "parsed": ok,
        "cut_at_max_tokens": cut,
        "refusal_hits": refusals,
        "tiny_replies": tiny,
        "by_rating": by_rating,
        "tokens": {k: spread(v) for k, v in tokens.items()},
        "words": {k: spread(v) for k, v in words.items()},
        "chars": {k: spread(v) for k, v in chars.items()},
        "engine": timing["engine"],
    }
    with open(f"{out_path}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
