"""The GBC program under KIR's reference interpreter: what one job means, with no machine.

``run(inputs, models)`` runs one image job; model nodes call ``models``
(``kohakugbc.nodes.Models``). The machine's results are checked against this.
"""

from kohakunode import parse
from kohakunode.run import Interpreter
from kohakunode.sig import build_env

from kohakugbc.nodes import library
from kohakugbc.program import ENTRY, text

INPUTS = ("image", "captions", "tags", "controls")


def run(inputs: dict, models, step_budget: int | None = None) -> dict:
    """The job's outputs (``graph`` JSON text, ``stats``) for ``inputs`` (``image``, and
    optionally ``captions``, ``tags``, ``controls``)."""
    program = parse(text())
    env = build_env(program, [library.build(models)])
    job = {k: inputs[k] for k in INPUTS if k in inputs}
    out = Interpreter(program, env, step_budget=step_budget).call(ENTRY, job)
    return {"graph": out["graph"], "stats": out["stats"]}
