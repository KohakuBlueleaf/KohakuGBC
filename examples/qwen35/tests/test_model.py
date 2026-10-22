"""The engine's forward against the checkpoint's reference forward (transformers), on a recorded
GBC request with its image: the prompt's last logits and every logit of the recorded reply fed
token by token. Resuming from a prefix snapshot equals a full prefill; a captured decode graph
equals the eager decode. Each check is shown able to fail."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from kohakugbc.nodes.specs import chat as specs
from qwen35.model.forward import Step
from qwen35.ops import Ops
from qwen35.serve.graphs import DecodeGraphs
from qwen35.serve.inputs import Preparer
from qwen35.serve.pools import Pools, plan
from qwen35.unit import PREFER

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

PAGE = 16
MAX_PAGES = 256
REPLY = 48


class Bench:
    """Sequences driven step by step over the engine's pools and ops."""

    def __init__(self, model, seqs=4, snapshots=2):
        sh = model.shape
        p = plan(sh, 3 * 2**30, seqs, snapshots, PAGE, torch.float32, model.dtype, 4096)
        assert not p.reasons, p.reasons
        self.model, self.dev = model, model.device
        self.pools = Pools(sh, p, model.device, torch.float32, model.dtype, snapshots)
        self.ops = Ops(PREFER)

    def open(self, tokens: int) -> dict:
        slot = self.pools.take_slot()
        self.pools.clear_slot(slot)
        return {"slot": slot, "table": self.pools.take_pages(-(-tokens // PAGE)), "cached": 0}

    def resume(self, snap, tokens: int) -> dict:
        slot = self.pools.take_slot()
        pages = self.pools.take_pages(-(-tokens // PAGE) - len(snap.pages))
        table = self.pools.resume(snap, slot, pages)
        return {"slot": slot, "table": table, "cached": snap.length}

    def rows(self, seq, start, end):
        t = np.arange(start, end)
        table = np.asarray(seq["table"])
        return torch.from_numpy(table[t // PAGE] * PAGE + t % PAGE)

    def step(self, parts, image=None) -> torch.Tensor:
        """``parts``: (seq, ids [k], positions [3, k]) per sequence; ``image``: (rows, embeds)."""
        cu, ids, pos, write, last, ctx = [0], [], [], [], [], []
        pages = torch.zeros(len(parts), MAX_PAGES, dtype=torch.int32)
        for i, (seq, t, p) in enumerate(parts):
            k = t.shape[0]
            ids.append(t)
            pos.append(p)
            write.append(self.rows(seq, seq["cached"], seq["cached"] + k))
            seq["cached"] += k
            ctx.append(seq["cached"])
            pages[i, : len(seq["table"])] = torch.tensor(seq["table"])
            cu.append(cu[-1] + k)
            last.append(cu[-1] - 1)
        slots = [s["slot"] for s, _, _ in parts]
        step = Step(
            ids=torch.cat(ids).to(self.dev),
            positions=torch.cat(pos, 1).to(self.dev),
            cu=torch.tensor(cu, dtype=torch.int32, device=self.dev),
            slots=torch.tensor(slots, dtype=torch.int32, device=self.dev),
            pages=pages.to(self.dev),
            ctx=torch.tensor(ctx, dtype=torch.int32, device=self.dev),
            write=torch.cat(write).to(self.dev),
            last=torch.tensor(last, device=self.dev),
            max_len=max(t.shape[0] for _, t, _ in parts),
            image_rows=None if image is None else image[0].to(self.dev),
            image_embeds=None if image is None else image[1],
        )
        return self.model.forward(self.ops, self.pools, step)


def kl(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Mean KL(ref || got) per row."""
    lr, lg = F.log_softmax(ref.float(), -1), F.log_softmax(got.float(), -1)
    return (lr.exp() * (lr - lg)).sum(-1).mean().item()


@pytest.fixture(scope="module")
def case(model, recorded):
    """A recorded request with an image, prepared; its recorded reply's tokens."""
    processor = AutoProcessor.from_pretrained(model.path)
    preparer = Preparer(
        processor, model.hf.model.get_rope_index, model.shape, specs.schema,
        boundaries=("system", "images"),
    )  # fmt: skip
    request, reply = next((r, t) for r, t in recorded if r["images"] and r["system"])
    prep = preparer(request)
    reply_ids = processor.tokenizer(reply, add_special_tokens=False)["input_ids"][:REPLY]
    return prep, torch.tensor(reply_ids)


@pytest.fixture(scope="module")
def reference_logits(model, case):
    """The transformers forward of prompt + reply: logits from the prompt's last token on."""
    prep, reply = case
    full = torch.cat([prep.ids, reply])[None].to(model.device)
    types = (full == model.shape.image_token).int()
    with torch.no_grad():
        out = model.hf(
            input_ids=full,
            pixel_values=prep.pixels.to(model.device),
            image_grid_thw=prep.grid.to(model.device),
            mm_token_type_ids=types,
        )
    model.hf.model.rope_deltas = None
    # A copy: a view would keep every position's logits alive
    return out.logits[0, len(prep.ids) - 1 :].float().clone()


def image_inputs(model, prep, start=0, end=None):
    end = len(prep.ids) if end is None else end
    embeds = model.vision(prep.pixels, prep.grid)
    image = (prep.ids == model.shape.image_token).nonzero().flatten()
    inside = (image >= start) & (image < end)
    return image[inside] - start, embeds[inside.nonzero().flatten().to(embeds.device)]


def run_ours(bench, model, prep, reply, image=True) -> torch.Tensor:
    seq = bench.open(len(prep.ids) + len(reply) + 1)
    images = image_inputs(model, prep) if image else None
    first = bench.step([(seq, prep.ids, prep.positions)], images)
    out = [first]
    position = prep.next_position
    for tok in reply[:-1]:
        p = torch.full((3, 1), position, dtype=torch.long)
        out.append(bench.step([(seq, tok[None], p)]))
        position += 1
    bench.pools.give_slot(seq["slot"])
    bench.pools.give_pages(seq["table"])
    return torch.cat(out)


def test_forward_matches_the_reference_model(model, case, reference_logits):
    prep, reply = case
    bench = Bench(model)
    ours = run_ours(bench, model, prep, reply)
    ref = reference_logits[: len(reply)]
    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.95, agree
    assert kl(ref, ours) < 0.02, kl(ref, ours)
    # The check sees a forward that drops the image
    blind = run_ours(bench, model, prep, reply, image=False)
    assert kl(ref, blind) > 10 * max(kl(ref, ours), 1e-3)


@pytest.mark.parametrize("boundary", [0, 1])
def test_resume_from_a_snapshot_equals_a_full_prefill(model, case, boundary):
    prep, reply = case
    bench = Bench(model)
    length, key = prep.boundaries[boundary]
    if boundary == 0:
        assert length % PAGE, "the system prompt should end inside a page (the private copy)"
    total = len(prep.ids) + len(reply) + 1

    full = bench.open(total)
    want = [bench.step([(full, prep.ids, prep.positions)], image_inputs(model, prep))]

    head = bench.open(total)
    images = image_inputs(model, prep, 0, length)
    bench.step([(head, prep.ids[:length], prep.positions[:, :length])], images)
    assert bench.pools.take_snapshot(key, length, head["slot"], head["table"])
    bench.pools.give_slot(head["slot"])
    bench.pools.give_pages(head["table"])

    snap = bench.pools.find(key)
    resumed = bench.resume(snap, total)
    tail = (prep.ids[length:], prep.positions[:, length:])
    got = [bench.step([(resumed, *tail)], image_inputs(model, prep, length))]

    broken = bench.resume(snap, total)
    bench.pools.conv[:, broken["slot"]].zero_()
    bad = bench.step([(broken, *tail)], image_inputs(model, prep, length))

    position = prep.next_position
    for tok in reply[:8]:
        p = torch.full((3, 1), position, dtype=torch.long)
        want.append(bench.step([(full, tok[None], p)]))
        got.append(bench.step([(resumed, tok[None], p)]))
        position += 1
    want, got = torch.cat(want), torch.cat(got)
    assert torch.equal(want.argmax(-1), got.argmax(-1))
    assert kl(want, got) < 2e-3, kl(want, got)
    assert kl(want[:1], bad) > 10 * max(kl(want[:1], got[:1]), 1e-4)


def test_decode_graph_equals_eager_decode(model, case):
    prep, reply = case
    bench = Bench(model)
    seqs = [bench.open(len(prep.ids) + 8) for _ in range(3)]
    images = _stacked_images(model, prep, len(seqs))
    bench.step([(s, prep.ids, prep.positions) for s in seqs], images)
    graphs = DecodeGraphs(model, bench.ops, bench.pools, 4, MAX_PAGES)
    with torch.inference_mode():
        graphs.capture()
    position = prep.next_position
    tok = reply[0]
    state = bench.pools.state.clone(), bench.pools.conv.clone()
    eager = bench.step([(s, tok[None], torch.full((3, 1), position)) for s in seqs])
    for s in seqs:
        s["cached"] -= 1
    bench.pools.state.copy_(state[0])
    bench.pools.conv.copy_(state[1])
    ids = torch.full((3,), int(tok), device=model.device)
    pos = torch.full((3, 3), position, device=model.device)
    slots = torch.tensor([s["slot"] for s in seqs], dtype=torch.int32, device=model.device)
    pages = torch.zeros(3, MAX_PAGES, dtype=torch.int32)
    for i, s in enumerate(seqs):
        pages[i, : len(s["table"])] = torch.tensor(s["table"])
    ctx = torch.tensor([s["cached"] + 1 for s in seqs], dtype=torch.int32, device=model.device)
    write = torch.cat([bench.rows(s, s["cached"], s["cached"] + 1) for s in seqs])
    graph = graphs.run(ids, pos, slots, pages.to(model.device), ctx, write.to(model.device))
    assert torch.allclose(graph, eager, atol=1e-2, rtol=1e-3)
    assert torch.equal(graph.argmax(-1), eager.argmax(-1))


def _stacked_images(model, prep, n):
    rows, embeds = image_inputs(model, prep)
    length = len(prep.ids)
    return (
        torch.cat([rows + i * length for i in range(n)]),
        torch.cat([embeds] * n),
    )
