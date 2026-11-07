# Evaluating predicted GBC graphs

`kohakugbc.eval` scores a predicted GBC graph against the ground truth. It runs two tracks:

- **Structure track**: uses boxes, vertex types and the graph (edges, and relations as vertex sets). It never reads text.
- **Full track**: the structure track plus text (names and descriptions), compared by embedding similarity, never by equality.

A prediction may use any vertex ids, any vertex order and any wording. The score is the best that
any one-to-one pairing of predicted with ground-truth vertices can reach, so none of these can
change it.

## What a release bundles

| file | role | needs |
|---|---|---|
| `src/kohakugbc/eval/spec.py` | a graph as the metrics see it; the JSON-lines reader | numpy |
| `src/kohakugbc/eval/match.py` | IoU and the correspondence program | numpy, scipy ≥ 1.9 (`milp`, HiGHS) |
| `src/kohakugbc/eval/score.py` | both tracks' scores and the diagnostics | numpy |
| `src/kohakugbc/eval/text.py` | the full track's text similarity | numpy (a sentence-transformers model is passed in) |
| `src/kohakugbc/eval/__init__.py` | the package's API | |
| `scripts/eval/score_graphs.py` | the command line | sentence-transformers, only with `--text-model` |
| `tests/eval/test_metrics.py` | the checks below; fixtures from `tests/fixtures/traces/nest/jobs.jsonl` | pytest |

The package imports only `numpy`, `scipy` and its own modules (`kohakugbc.eval.*`). To ship it
without the rest of KohakuGBC, copy the folder as `kohakugbc/eval/` beside an empty
`kohakugbc/__init__.py`.

The ground truth must be the repaired graphs (`scripts/data/repair_graphs.py`):

- every relation names vertices that exist;
- the graph is acyclic;
- the masks follow from the boxes.

## Running it

```
python scripts/eval/score_graphs.py GT.jsonl PRED.jsonl --out report.json
    [--rows per_image.jsonl] [--taus 0.5 0.55 ...] [--text-model google/embeddinggemma-300m]
```

Each line is a GBC graph, or `{"key": ..., "graph": <graph or its JSON text>}`. An image is
keyed by `key`, else by the graph's `img_path`.

- A ground-truth image with no prediction is scored against an empty graph.
- Predictions for images not in the ground truth are counted (`unknown`) and ignored.
- The structure track always runs. `--text-model` adds the full track; its baselines are measured on the ground truth and written to the report.

From Python:

```python
from kohakugbc.eval import evaluate, read
from kohakugbc.eval.spec import empty

gt, pred = read("GT.jsonl"), read("PRED.jsonl")
report, rows = evaluate((k, pred.get(k, empty()), g) for k, g in gt.items())
print(report["structure"]["score"])
```

### What a prediction must hold

| field | required | meaning |
|---|---|---|
| `vertex_id` | yes | any string unique in the graph |
| `label` | yes | `image`, `entity`, `composition` or `relation` |
| `bbox` | yes, except on `image` and `relation` | GBC's `{"left", "top", "right", "bottom", "confidence"}` or `[l, t, r, b]` / `[l, t, r, b, score]`, normalized to [0, 1] |
| `out_edges` | yes | `[{"target": vertex_id, ...}]`; a relation vertex's out-edges name its members |
| `out_edges[].text` | full track | the child's name, as written in the parent's caption |
| `descs` | full track | `[{"text": ...}]` or strings, on vertices, relations and the image |
| `name` | full track | an extra name for a vertex |
| box confidence | diagnostic | orders vertices for vertex AP; missing counts as 1 |

## The formulation

### What is compared

A graph is reduced to `G = (V, b, c, E, R)`:

- `V`: the vertices other than the image and the relation vertices. Each has a box `b(v)` and a type `c(v) ∈ {entity, composition}`.
- `E ⊆ (V ∪ {img}) × V`: the edges, parent to child. An edge from the image marks a top-level vertex. Edges into the image and self-loops are dropped.
- `R`: each relation as the set of vertices its out-edges name. A set named twice counts as one relation. A relation's box is the union of its members' boxes and is not scored.
- Dropped: vertex ids, vertex order, sub/super masks (these follow from the boxes) and, in the structure track, all text.

The prediction is `Ĝ = (V̂, b̂, ĉ, Ê, R̂)`.

### Correspondences

A correspondence `m` is a partial one-to-one map `V̂ → V`, with `m(img) = img`. At an IoU
threshold `τ`, a pair is allowed when

```
m(u) = v  only if  IoU(b̂(u), b(v)) ≥ τ  and  ĉ(u) = c(v).
```

Under `m`:

```
TP_V(m) = |dom m|
TP_E(m) = |{(a, b) ∈ Ê : (m(a), m(b)) ∈ E}|
TP_R(m) = |{r ∈ R̂ : r ⊆ dom m and m(r) ∈ R}|
F_X(m)  = 2 TP_X(m) / (|X̂| + |X|)        X ∈ {V, E, R}
```

A part is **active** when `|X̂| + |X| > 0`. Means below run over the active parts only, and are 1
when no part is active.

### Structure track

```
S_τ(Ĝ, G)    = max over allowed m of  mean_{X active} F_X(m)
image score  = mean of S_τ over τ ∈ {0.50, 0.55, ..., 0.95}
dataset score = mean of image scores
```

- **Invariance.** Renaming or reordering the predicted vertices relabels the set of allowed correspondences one-to-one and leaves every count unchanged, so the maximum is unchanged.
- **Twins.** Two vertices with the same box and type are told apart by their edges and relations. If they are truly identical, either pairing gives the same score.
- **Endpoints.** The identity correspondence gives a prediction equal to the ground truth a score of 1. An empty prediction scores 0.
- **Box quality.** The τ sweep prices box quality: a vertex at IoU 0.8 can be paired at 7 of the 10 thresholds. Edges and relations count only where their ends are paired.

### Full track

Texts are compared by meaning: a vision-language model names the same vertex differently from
run to run. Two texts score the cosine of their unit sentence embeddings, rescaled so that
unrelated text scores 0:

```
s(a, b) = clip((cos(a, b) − β) / (1 − β), 0, 1)
```

`β` is the mean cosine between a text of one ground-truth image and a text of the same kind
from another: 2,000 random pairs, seed 0, measured per kind (names, vertex descriptions,
relation descriptions, image descriptions).

- Names of a vertex (the texts of the edges into it, and `name`): `s_N` = the best pair.
- Descriptions (a vertex's, a relation's, the image's): `s_D` = greedy F over sentences, as BERTScore does over tokens.
  - `P` is the mean, over predicted sentences, of their best ground-truth match.
  - `R` is the same taken the other way.
  - `s_D = 2PR / (P + R)`.
- A side without text scores 0.

Similarity takes the place of a hit, on the same denominators:

```
F_N(m) = 2 Σ_{u ∈ dom m} s_N(u, m(u)) / (|V̂| + |V|)
F_D(m) = 2 Σ_{u ∈ dom m} s_D(u, m(u)) / (|V̂| + |V|)
F_Q(m) = 2 Σ_{r hit}     s_D(r, m(r)) / (|R̂| + |R|)        relation descriptions
F_C    = s_D(image, image)                                 counted once per side that has any

S_τ^full = max over allowed m of  ½ (mean_active{F_V, F_E, F_R} + mean_active{F_N, F_D, F_Q, F_C})
```

The full track picks its own maximizing correspondence. Text can choose between pairings the boxes
allow, but cannot create a pair the boxes forbid. A prediction with perfect structure and no
text scores exactly 0.5.

### Computing the maximum

`|X̂| + |X|` does not depend on `m`, so each track's score is linear in the hits. Its maximum is
an integer program, solved exactly with HiGHS (`scipy.optimize.milp`, `mip_rel_gap = 0`):

```
maximize   Σ wV[u,v] x[u,v]  +  wE Σ y[e,f]  +  Σ wR[r,s] z[r,s]
x[u,v]     one per allowed pair;  Σ_v x[u,v] ≤ 1,  Σ_u x[u,v] ≤ 1
y[e,f]     predicted edge e = (a,b) onto ground-truth edge f = (c,d):
           y ≤ x[b,d],  y ≤ x[a,c]  (the second dropped when a and c are the image)
z[r,s]     |r| = |s|:  z ≤ Σ_{v ∈ s} x[u,v]  for every u ∈ r   (with one-to-one x, m(r) = s)
```

- **Weights.** With `k` active parts, `wX = 2 / (k (|X̂| + |X|))`.
- **Full track.** The structure weights are halved, and each active text part adds `2 / (2 k_T (|X̂| + |X|))` times its similarity to `wV` or `wR`.
- **Ties.** Ties break toward the larger total IoU (an added `1e-7 · IoU / #pairs`). This fixes one correspondence for the diagnostics.
- **Skipping the solver.** When no vertex has more than one allowed partner, every allowed pair is taken without solving, since all weights are positive.

The score is recomputed from the chosen `m` by counting, independently of the program.

## The report

```
{
  "options": {"taus": [...]},
  "images": n, "missing": n, "unknown": n,
  "structure": {
    "score": dataset score,
    "taus": {"0.5": {
        "score": mean over images of S_τ,
        "vertex" | "edge" | "relation": {"precision", "recall", "f1"},
        "types": {"entity" | "composition": {"precision", "recall", "f1"}},
        "counts": {part: [TP, predicted, ground truth], ...},
        "vertex_ap": AP
    }, ...}
  },
  "full": {... the same, plus "names", "descs", "relation_descs", "image_descs"},
  "text_model": ..., "text_baselines": {kind: β}
}
```

- The per-τ precision/recall/F1 are **diagnostics**: counts summed over images (micro) under each track's chosen correspondence. The **score** is the per-image maximum averaged over images.
- Vertex AP ranks predicted vertices by box confidence over the whole dataset, with COCO's greedy matching per type and 101-point interpolation, averaged over types.
- `--rows` writes one line per image: each track's image score and, per τ, its score and counts.

## Checks

`tests/eval/test_metrics.py`:

- **The maximum is exact.** On 60 random small graphs with shared boxes (so many pairs tie), the program finds the same best score as brute force over every allowed correspondence, in both tracks.
- **Order, ids and text never matter.** A graph scores 1 against itself shuffled, renamed and stripped of text, and so do twin vertices listed in any order.
- **Each break moves its own part.**
  - Dropping vertices costs exactly their share of vertex recall.
  - Rewired edges cost edges but not vertices.
  - A lost relation costs exactly its share of relation recall.
  - Box noise lowers the score monotonically.
  - A type mismatch prevents a pair.
- **Text is read by meaning.** Synonymous names score 1, unrelated names score lower, and the structure track ignores both.

On 3,000 repaired web1m graphs (about 9 vertices each):

- Shuffled, renamed and text-stripped predictions give the same score on every image (largest difference 0.0).
- 5% box jitter scores 0.727 (1.000 at τ = 0.5, 0.891 at 0.75, 0.047 at 0.95).
- Perfect boxes with every vertex hung off the image score 0.547.
- The structure track takes about 28 ms per image over all ten thresholds.
