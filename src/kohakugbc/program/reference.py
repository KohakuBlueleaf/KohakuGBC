"""The GBC program under KIR's reference interpreter: what one job means, with no machine.

``run(inputs, models)`` runs one image job; model nodes call ``models``
(``kohakugbc.nodes.Models``). The machine's results are checked against this.
"""

from kohakunode import NodeLibrary, parse
from kohakunode.run import Interpreter
from kohakunode.sig import build_env

from kohakugbc.nodes import library
from kohakugbc.program import ENTRY, text

CALL = '@use "job"\n'


def run(inputs: dict, models, step_budget: int | None = None) -> dict:
    """The job's outputs (``graph`` JSON text, ``stats``) for ``inputs`` (``image``, and
    optionally ``captions``, ``tags``, ``controls``)."""
    job = NodeLibrary("job")

    @job.node(outputs=["image", "captions", "tags", "controls"])
    def job_inputs():
        return (
            inputs["image"],
            inputs.get("captions"),
            inputs.get("tags"),
            inputs.get("controls"),
        )

    source = (
        CALL
        + text()
        + "\n()job_inputs(image, captions, tags, controls)\n"
        + f"(image, captions, tags, controls){ENTRY}(graph, stats)\n"
    )
    program = parse(source)
    env = build_env(program, [library.build(models), job])
    out = Interpreter(program, env, step_budget=step_budget).run()
    return {"graph": out["graph"], "stats": out["stats"]}
