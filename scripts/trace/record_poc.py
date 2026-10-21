"""Record one PoC run as a trace (``kohakugbc.nodes.specs.trace``).

Runs ``scripts/run_iterative.py`` of the PoC (``poc/gbc-poc``) in this process with its own
arguments, in the PoC's environment, and writes every VLM request and reply, every detector call
with its raw and selected boxes, every embedding and same-object decision, and each image's
final records and graph, keyed the way the GBC program asks.

The PoC's unseeded random choices (the relation count in the relation prompt, the phrase that
names a set of children, the composition hints) and its set-ordered relation sentences are
replaced by the program's rules (``kohakugbc.nodes.specs.draw``, first-seen order), so the
recorded run is one the program can reproduce.

    <poc venv>/python scripts/trace/record_poc.py --trace runs/traces/text --poc poc/gbc-poc \
        -- --config configs/local_v3_json.yaml ... --run_name trace_text
"""

import argparse
import json
import os
import re
import runpy
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import gbcpoc.engine.runner as runner  # noqa: E402
import gbcpoc.texts as texts  # noqa: E402
import gbcpoc.vlm as gbcpoc_vlm  # noqa: E402
import numpy as np  # noqa: E402
from gbc.captioning.detection import detection as gbc_detection  # noqa: E402
from gbc.captioning.detection.detection_action import DetectionAction  # noqa: E402
from gbc.captioning.mllm import composition_mst, parse_mllm_output  # noqa: E402
from gbc.captioning.mllm.query_prototype import RelationQuery  # noqa: E402
from gbc.captioning.pipeline import get_relational_queries as grq  # noqa: E402
from gbc.captioning.primitives import ActionOutput  # noqa: E402
from gbcpoc.detection.base import BatchedTextDetection  # noqa: E402

from kohakugbc.graph.records import from_node_info  # noqa: E402
from kohakugbc.nodes.specs import draw  # noqa: E402
from kohakugbc.nodes.specs import trace as tr  # noqa: E402

LOCAL = threading.local()
WRITER: tr.TraceWriter | None = None
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
EMBEDDED: set[str] = set()


def img_key(path: str) -> str:
    return os.path.basename(path)


def spec_of(action) -> str:
    """The prompt set of a query action: the folder of its prompt files."""
    return Path(action.system_file or action.query_file).parent.name


# ---- determinism: the program's rules in place of the PoC's random choices -------------------


def extract_relation_sentences(text: str) -> list[str]:
    """ml-gbc's pattern; repeated sentences dropped in first-seen order (not set order)."""
    pattern = re.compile(r"(?:^|\n)-\s*([^\n]+)|(?:^|\n)\d+\.\s*([^\n]+)")
    return list(dict.fromkeys(m[0] or m[1] for m in pattern.findall(text)))


def relation_query(self, action_input, queried_nodes=None):
    """ml-gbc's RelationQuery.query with the relation count drawn by the program's rule."""
    entities = action_input.entities
    if len(entities) < 2:
        return ActionOutput([], None, None)
    to_relate = ", ".join(f"[{e.text}]" for e in entities)
    image = img_key(action_input.img_path)
    count = draw.rng("relation-count", image, action_input.first_entity_id).randint(
        2, len(entities)
    )
    pil = action_input.get_image(return_pil=True)
    query_output = self.query_prelim(pil, (count, to_relate))
    query_result = self.parse_output_func(query_output, action_input)
    self._add_to_queried_nodes(query_result, action_input, queried_nodes)
    return ActionOutput([], query_result, pil)


class _TextChooser:
    """``rng.choice(texts)`` of get_relational_queries, drawn per vertex and text set."""

    def __init__(self, image: str, vertex_id: str):
        self.image, self.vertex_id = image, vertex_id

    def choice(self, texts):
        return draw.rng("relational-text", self.image, self.vertex_id, *texts).choice(list(texts))


_per_vertex = grq.RelationalQueryManager._get_relational_queries_for_vertex


def per_vertex(self, vertex, rng):
    return _per_vertex(self, vertex, _TextChooser(img_key(self.img_path), vertex.vertex_id))


class _MstNumpy:
    """numpy for composition_mst, with ``np.random.random`` from the current composition's rng."""

    class _Random:
        @staticmethod
        def random():
            return LOCAL.mst_rng.random()

    random = _Random()

    def __getattr__(self, name):
        return getattr(np, name)


def mst_choice(seq):
    return LOCAL.mst_rng.choice(seq)


def wrap_composition(cls):
    original = cls.query

    def query(self, action_input, queried_nodes=None):
        infos = action_input.entity_info
        first = (infos[0] if isinstance(infos, list) else infos).entity_id
        LOCAL.mst_rng = draw.rng("composition", img_key(action_input.img_path), first)
        return original(self, action_input, queried_nodes)

    cls.query = query


# ---- VLM requests ------------------------------------------------------------------------


def vlm_key(meta: dict) -> str | None:
    """The trace key of the request being logged, from the action and input it was made for."""
    action = meta["action"].split("+")[0]
    stash = getattr(LOCAL, "query", None)
    match action:
        case "OpenRouterImageQuery":
            act, inp = stash
            return tr.vlm_key("image_query", spec_of(act), img_key(inp.img_path))
        case "OpenRouterEntityQuery":
            act, inp = stash
            return tr.vlm_key(
                "entity",
                spec_of(act),
                img_key(inp.img_path),
                entity=inp.first_entity_id,
                box=tr.box_key(None if inp.bbox is None else inp.bbox.to_xyxy()),
            )
        case "OpenRouterRelationQuery":
            act, inp = stash
            return tr.vlm_key(
                "relation",
                spec_of(act),
                img_key(inp.img_path),
                vertex=inp.first_entity_id,
                box=tr.box_key(inp.bbox.to_xyxy()),
                entities=[[e.entity_id, e.text] for e in inp.entities],
            )
        case "OpenRouterCompositionQuery":
            act, inp = stash
            return tr.vlm_key(
                "composition",
                spec_of(act),
                img_key(inp.img_path),
                entity=[
                    e.entity_id
                    for e in (
                        inp.entity_info if isinstance(inp.entity_info, list) else [inp.entity_info]
                    )
                ],
                parts=[[e.entity_id, e.text] for e in inp.entities],
                boxes=[tr.box_key(b.to_xyxy()) for b in inp.bboxes],
            )
        case "EntityBatchQuery":
            boxes = LOCAL.batch
            ids = meta["entity_id"].split(",")
            return tr.vlm_key(
                "entity_batch",
                "json_batch",
                img_key(meta["img_path"]),
                regions=[[i, boxes[i]] for i in ids],
            )
        case "CoverageQuery":
            return tr.vlm_key(
                "coverage", "coverage", img_key(meta["img_path"]), labels=LOCAL.labels
            )
        case a if a.startswith("SeedElements:"):
            return tr.vlm_key(
                "seed_elements",
                "seeded",
                img_key(meta["img_path"]),
                source=a.split(":", 1)[1],
                tag_hint=ARGS.seed_tag_hint,
            )
    return None


def wrap_vlm():
    client = gbcpoc_vlm.openrouter.VlmClient
    write, chat = client._write, client.chat

    def chat_recorded(self, image, query, system_message, log_meta, **kwargs):
        LOCAL.request = {
            "system": system_message,
            "query": query,
            "schema": kwargs.get("json_schema_file") and Path(kwargs["json_schema_file"]).name,
            "max_tokens": kwargs.get("max_tokens"),
        }
        try:
            return chat(self, image, query, system_message, log_meta, **kwargs)
        finally:
            LOCAL.request = None

    def write_recorded(self, record):
        write(self, record)
        request = dict(getattr(LOCAL, "request", None) or {"query": record.get("query")})
        request.update(images=record.get("sent_size"), sampling=record.get("sampling"))
        WRITER.write(
            "vlm",
            {
                "key": vlm_key(record),
                "action": record.get("action"),
                "prefill": record.get("prefill"),
                "request": request,
                "reply": record.get("output", ""),
                "error": record.get("error"),
                "latency": record.get("latency"),
                "ttft": record.get("ttft"),
                "usage": record.get("usage"),
                "finish_reason": record.get("finish_reason"),
            },
        )

    client.chat, client._write = chat_recorded, write_recorded

    action = gbcpoc_vlm.openrouter.OpenRouterQueryAction
    prelim = action.query_prelim

    def query_prelim(self, image, filled_in_query=None):
        LOCAL.query = (self, gbcpoc_vlm.openrouter.current_input(self))
        try:
            return prelim(self, image, filled_in_query)
        finally:
            LOCAL.query = None

    action.query_prelim = query_prelim


# ---- detector, embeddings, same-object ---------------------------------------------------------


def wrap_detection():
    query, detect = DetectionAction.query, gbc_detection.Detection.detect
    core = BatchedTextDetection.detect_core

    def action_query(self, action_input, queried_nodes=None):
        LOCAL.detect_input = action_input
        try:
            return query(self, action_input, queried_nodes)
        finally:
            LOCAL.detect_input = None

    def detect_core(self, image, texts):
        start = time.time()
        out = core(self, image, texts)
        LOCAL.detect_raw = (out, [int(image.shape[1]), int(image.shape[0])], time.time() - start)
        return out

    def detect_recorded(self, image, texts):
        selected = detect(self, image, texts)
        inp = LOCAL.detect_input
        (boxes, scores, labels), size, latency = LOCAL.detect_raw
        names = [t if isinstance(t, str) else t.text for t in texts]
        WRITER.write(
            "detect",
            {
                "key": tr.detect_key(
                    img_key(inp.img_path),
                    None if inp.bbox is None else inp.bbox.to_xyxy(),
                    names,
                ),
                "size": size,
                "params": [
                    {
                        "nms": t.nms_threshold,
                        "high_conf": t.select_high_conf_tolerance,
                        "max_rel_area": t.max_rel_area,
                        "topk": t.topk,
                    }
                    for t in texts
                    if not isinstance(t, str)
                ],
                "raw": {
                    "boxes": [[float(v) for v in b] for b in boxes],
                    "scores": [float(s) for s in scores],
                    "labels": [int(lab) for lab in labels],
                },
                "selected": {
                    text: [[b.left, b.top, b.right, b.bottom, b.confidence] for b in bs]
                    for text, bs in selected.items()
                },
                "latency": latency,
            },
        )
        return selected

    DetectionAction.query = action_query
    BatchedTextDetection.detect_core = detect_core
    gbc_detection.Detection.detect = detect_recorded


def wrap_texts():
    embed, same = texts._embed, texts.embedding_same_object

    def embed_recorded(model_name, device, text):
        vector = embed(model_name, device, text)
        with JOBS_LOCK:
            new = text not in EMBEDDED
            EMBEDDED.add(text)
        if new:
            WRITER.write("embed", {"text": text, "vector": tr.pack(vector)})
        return vector

    def same_recorded(text1, text2, **kwargs):
        result = same(text1, text2, **kwargs)
        score = None
        if kwargs.get("method", "embedding") == "embedding":
            model = kwargs.get("model_name", "google/embeddinggemma-300m")
            device = kwargs.get("device", "cpu")
            score = float(embed(model, device, text1) @ embed(model, device, text2))
        WRITER.write("same", {"a": text1, "b": text2, "same": bool(result), "score": score})
        return result

    texts._embed, texts.embedding_same_object = embed_recorded, same_recorded


# ---- engine: rounds, labels, batches, final records and graph ------------------------------------


def wrap_engine():
    engine = runner.RoundEngine
    step, breadth, prefill = engine.step, engine.breadth_round, engine._prefill_batch
    convert = runner.node_infos_to_gbc_graphs

    def step_recorded(self, state):
        action = step(self, state)
        with JOBS_LOCK:
            JOBS.setdefault(img_key(state.img_path), {}).setdefault("rounds", []).append(action)
        return action

    def breadth_recorded(self, state):
        LOCAL.labels = state.labels()
        return breadth(self, state)

    def prefill_recorded(self, pairs):
        LOCAL.batch = {
            p.action_input.first_entity_id: tr.box_key(p.action_input.bbox.to_xyxy()) for p in pairs
        }
        return prefill(self, pairs)

    def convert_recorded(node_infos, *args, **kwargs):
        graphs = convert(node_infos, *args, **kwargs)
        if node_infos:
            key = img_key(node_infos[0].img_path)
            dumps = [n.model_dump() for n in node_infos]
            with JOBS_LOCK:
                job = JOBS.setdefault(key, {})
                job["records"] = [from_node_info(json.loads(json.dumps(d))) for d in dumps]
                job["graph"] = graphs[0].model_dump() if graphs else None
        return graphs

    engine.step, engine.breadth_round, engine._prefill_batch = (
        step_recorded,
        breadth_recorded,
        prefill_recorded,
    )
    runner.node_infos_to_gbc_graphs = convert_recorded


# ---- main -------------------------------------------------------------------------------------


def parse_run_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seeds", nargs="*", default=[])
    p.add_argument("--image_list", default=None)
    p.add_argument("--seed_elements", default="text")
    p.add_argument("--seed_tag_hint", action="store_true")
    p.add_argument("--controls", nargs="*", action="extend", default=[])
    p.add_argument("--run_name")
    p.add_argument("--prompt_dir", default=None)
    args, _ = p.parse_known_args(argv)
    return args


def job_inputs(poc: Path, args: argparse.Namespace) -> dict[str, dict]:
    """Each image's job inputs: the seed captions and tags of seeded arms."""
    inputs = {}
    for path in args.seeds:
        with open(poc / path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seed = json.loads(line)
                    inputs[img_key(seed["image"])] = {
                        "captions": {
                            "detail": seed.get("description"),
                            "short": seed.get("brief"),
                            "title": seed.get("title"),
                        },
                        "tags": seed.get("tags"),
                        "source": seed.get("source"),
                    }
    return inputs


ARGS: argparse.Namespace


def main() -> int:
    global WRITER, ARGS
    cut = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--poc", required=True)
    parser.add_argument("--arm", required=True, help="the arm's name in the trace meta")
    ours = parser.parse_args(sys.argv[1:cut])
    run_argv = sys.argv[cut + 1 :]
    ARGS = parse_run_args(run_argv)
    poc, trace_dir = Path(ours.poc).resolve(), Path(ours.trace).resolve()

    # run_iterative's arguments name files relative to the PoC root
    os.chdir(poc)
    WRITER = tr.TraceWriter(
        trace_dir,
        {
            "source": "poc/gbc-poc scripts/run_iterative.py",
            "arm": ours.arm,
            "argv": run_argv,
            "image_root": str((poc / "data" / "seeds" / "images").resolve()).replace("\\", "/"),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    parse_mllm_output.extract_relation_sentences = extract_relation_sentences
    RelationQuery.query = relation_query
    grq.RelationalQueryManager._get_relational_queries_for_vertex = per_vertex
    composition_mst.np = _MstNumpy()
    composition_mst.choice = mst_choice
    wrap_composition(gbcpoc_vlm.openrouter.OpenRouterCompositionQuery)
    wrap_vlm()
    wrap_detection()
    wrap_texts()
    wrap_engine()

    sys.argv = ["run_iterative.py", *run_argv]
    try:
        runpy.run_path(str(poc / "scripts" / "run_iterative.py"), run_name="__main__")
    finally:
        inputs = job_inputs(poc, ARGS)
        controls = dict(c.split("=", 1) for c in ARGS.controls)
        for key, job in sorted(JOBS.items()):
            WRITER.write(
                "jobs",
                {
                    "key": key,
                    "image": key,
                    "inputs": {**inputs.get(key, {}), "controls": controls},
                    "rounds": job.get("rounds", []),
                    "records": job.get("records", []),
                    "graph": job.get("graph"),
                },
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
