"""Round nodes: the frontier, the node budget, what the job does next.

Ports of the PoC's ``RoundEngine.step`` and ``depth_round``. The frontier is a priority list
(``priority``: shallow and large first by default; ties in insertion order); a round works on
the frontier as it was when the round began, in waves that take what the budget still needs.
"""

from kohakugbc.nodes.registry import node


def described(records: list) -> int:
    """Image and entity vertices so far: records with descriptions."""
    return sum(1 for r in records if r["descs"])


def _order(region: dict, controls: dict) -> tuple:
    b = region["box"]
    area = 1.0 if b is None else max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    match controls["priority"]:
        case "area":
            return (-area, region["order"])
        case "deep":
            return (-region["depth"], -area, region["order"])
        case _:
            return (region["depth"], -area, region["order"])


@node(outputs=["frontier"])
def empty_frontier() -> dict:
    return {"regions": [], "next": 0}


@node(outputs=["frontier"])
def push(frontier: dict, found: list) -> dict:
    """The found regions join the frontier, numbered in the order they came."""
    n = frontier["next"]
    added = [{**r, "order": n + k} for k, r in enumerate(found)]
    return {"regions": [*frontier["regions"], *added], "next": n + len(added)}


@node(outputs=["step"])
def next_step(records: list, frontier: dict, rounds: int, breadth: int, stop, controls: dict):
    """``depth`` while regions wait and the budget allows; ``breadth`` below the minimum while
    coverage rounds remain; else ``done``."""
    if stop or rounds >= controls["max_rounds"]:
        return "done"
    n = described(records)
    if n >= controls["max_nodes"]:
        return "done"
    if frontier["regions"]:
        return "depth"
    if n < controls["min_nodes"] and breadth < controls["max_breadth_rounds"]:
        return "breadth"
    return "done"


@node(outputs=["current", "frontier", "limit"])
def round_start(frontier: dict, records: list, controls: dict) -> tuple[list, dict, int]:
    """The round's regions (the whole frontier, in priority order), a fresh frontier for the
    regions found now, and the vertex count the round stops at."""
    current = sorted(frontier["regions"], key=lambda r: _order(r, controls))
    n = described(records)
    budget = controls["max_nodes"] - n
    if controls["round_size"] is not None:
        budget = min(budget, controls["round_size"])
    return current, {"regions": [], "next": frontier["next"]}, n + budget


@node(outputs=["more"])
def needs_more(current: list, records: list, limit: int) -> bool:
    return bool(current) and described(records) < limit


@node(outputs=["picked", "current"])
def take(current: list, records: list, limit: int, controls: dict) -> tuple[list, list]:
    """One wave: as many regions as the budget still needs (at most ``entity_batch``)."""
    n = max(1, min(controls["entity_batch"], limit - described(records), len(current)))
    return current[:n], current[n:]


@node(outputs=["frontier"])
def requeue(frontier: dict, current: list) -> dict:
    """Regions the round did not reach wait for the next one, keeping their order."""
    return {"regions": [*current, *frontier["regions"]], "next": frontier["next"]}


@node(outputs=["when"])
def relate_mode(controls: dict) -> str:
    return controls["relate"]


@node(outputs=["reply"])
def no_reply(err) -> str:
    """A request that failed answers with an empty reply, as GBC reads one."""
    return ""
