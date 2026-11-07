"""Scores of a predicted GBC graph against the ground truth, in two tracks.

- Structure track: boxes, vertex types and the graph only (edges, relations as vertex sets).
- Full track: the structure track and the text (names, descriptions), compared by embedding
  similarity, never by equality.

A track's score at an IoU threshold is the best its correspondence can make it: the
correspondence between predicted and ground-truth vertices is chosen to maximize the score
(an exact integer program), so ids, names and the order of vertices never enter it.

- ``spec``: a graph as the metrics see it.
- ``match``: the correspondence program.
- ``score``: both tracks' scores and the diagnostics.
- ``text``: the full track's text similarity.

``scripts/eval/score_graphs.py`` scores a prediction file against the ground truth.
"""

from kohakugbc.eval.match import iou
from kohakugbc.eval.score import DEFAULT, TAUS, Options, correspond, evaluate, measure
from kohakugbc.eval.spec import IMAGE, Spec, empty, load, read

__all__ = [
    "DEFAULT",
    "IMAGE",
    "TAUS",
    "Options",
    "Spec",
    "correspond",
    "empty",
    "evaluate",
    "iou",
    "load",
    "measure",
    "read",
]
