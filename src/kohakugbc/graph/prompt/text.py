"""The prompt DAG as text, and text back as the DAG.

    <gbc bbox=yxyx space=abs_centralized canvas=1024x1024>     the header (coords.py)
    # [image] text                          the root
    tags: a, b                              tags of the item above (one level deeper)
    - [handle] [y0,x0,y1,x1] text           a vertex; box and text optional
      - [child] ...                         nested: an edge from the item above
    - relation [a] [b]: text                a relation of the item above

One handle in several places is one vertex, each place a parent edge; its parts may sit at
any place (given twice, they must agree). Relation participants: the handles before the colon
and in the text. ``parse(render(d))`` is ``d`` with boxes at ``Coords.snap`` (written as ints),
and ``render(parse(t)) == t`` for any ``t`` that ``render`` wrote.
"""

import re

from kohakugbc.graph.prompt.coords import HEADER, MOSS, Coords
from kohakugbc.graph.prompt.dag import BOX_TEXT, MENTION, Dag, DagError, Relation, Vertex, check

INDENT = "  "
HEAD = re.compile(rf"\[([^\[\]\n]+)\](?: {BOX_TEXT})?(?: (.+))?")
RELATION = re.compile(r"relation((?: \[[^\[\]\n]+\])*):(?: (.+))?")
TAGS = re.compile(r"tags:(?: (.+))?")
PLACEMENTS = ("first", "last")


def _head(handle: str, v: Vertex, coords: Coords) -> str:
    box = "" if v.box is None else " [{},{},{},{}]".format(*coords.write(v.box))
    return f"[{handle}]{box}" + ("" if v.text is None else f" {v.text}")


def _tags(tags: list[str]) -> str:
    return "tags:" + (" " + ", ".join(tags) if tags else "")


def _lines(
    dag: Dag, host: dict[str, str], coords: Coords
) -> tuple[list[str], dict[str, list[str]]]:
    """The text lines with each vertex's content under ``host[vertex]``, and every vertex's
    parents in the order its places appear."""
    vs = dag.vertices
    root = vs[dag.root]
    lines = [f"# {_head(dag.root, root, coords)}"]
    if root.tags is not None:
        lines.append(_tags(root.tags))
    places: dict[str, list[str]] = {}

    def walk(h: str, depth: int) -> None:
        pad = INDENT * depth
        for c in vs[h].children:
            places.setdefault(c, []).append(h)
            if host[c] != h:
                lines.append(f"{pad}- [{c}]")
                continue
            lines.append(f"{pad}- {_head(c, vs[c], coords)}")
            if vs[c].tags is not None:
                lines.append(f"{pad}{INDENT}{_tags(vs[c].tags)}")
            walk(c, depth + 1)
        for r in vs[h].relations:
            who = "".join(f" [{p}]" for p in r.participants)
            lines.append(f"{pad}- relation{who}:" + (f" {r.text}" if r.text else ""))

    walk(dag.root, 0)
    return lines, places


def hosts(dag: Dag, placement: str = "first") -> dict[str, str]:
    """The parent each vertex's content sits under: its ``first`` or ``last`` place in the text."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement {placement!r}: one of {PLACEMENTS}")

    # "first": the parent a depth-first walk reaches the vertex from first
    host: dict[str, str] = {}

    def first(h: str) -> None:
        for c in dag.vertices[h].children:
            if c not in host:
                host[c] = h
                first(c)

    first(dag.root)
    if placement == "first":
        return host

    # "last": moving a vertex's content moves only its own subtree, so re-placing at the last
    # place until nothing moves settles; any host is a real parent, so every round is valid
    for _ in range(4 * len(dag.vertices) + 1):
        _, places = _lines(dag, host, MOSS)
        moved = {c: ps[-1] for c, ps in places.items() if ps[-1] != host[c]}
        if not moved:
            break
        host.update(moved)
    return host


def render(dag: Dag, coords: Coords = MOSS, placement: str = "first", header: bool = True) -> str:
    """The text of ``dag``, boxes in ``coords``; ``header``: open with the line declaring them."""
    check(dag)
    lines, _ = _lines(dag, hosts(dag, placement), coords)
    return "\n".join(([coords.header()] if header else []) + lines) + "\n"


def _merge(handle: str, v: Vertex, text, box, tags) -> None:
    for name, new in (("text", text), ("box", box), ("tags", tags)):
        if new is None:
            continue
        old = getattr(v, name)
        if old is not None and old != new:
            raise DagError(f"[{handle}]: {name} given twice and differs: {old!r} / {new!r}")
        setattr(v, name, new)


def _parts(m: re.Match, coords: Coords, n: int) -> tuple[str, str | None, tuple | None]:
    try:
        box = None if m.group(2) is None else coords.read([int(m.group(i)) for i in range(2, 6)])
    except ValueError as e:
        raise DagError(f"line {n}: {e}") from None
    return m.group(1), m.group(6), box


def parse(text: str, coords: Coords = MOSS) -> Dag:
    """The DAG a prompt text says, its boxes read in the header's coords (else ``coords``);
    ``DagError`` names the first line it cannot read."""
    lines = [(n, line.rstrip()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if lines and HEADER.fullmatch(lines[0][1]):
        try:
            coords = Coords.from_header(lines[0][1])
        except ValueError as e:
            raise DagError(f"line {lines[0][0]}: {e}") from None
        lines = lines[1:]
    if not lines or not lines[0][1].startswith("# "):
        raise DagError("a prompt starts with the root, `# [handle] ...`")
    n, first = lines[0]
    m = HEAD.fullmatch(first[2:])
    if m is None:
        raise DagError(f"line {n}: the root is `# [handle] [a,b,c,d] text`")
    root, rtext, rbox = _parts(m, coords, n)
    vs = {root: Vertex(text=rtext, box=rbox)}
    parents = [root]  # parents[d]: the vertex items at depth d nest under
    last = (root, -1)  # the latest item, for a tags line under it
    for n, line in lines[1:]:
        body = line.lstrip(" ")
        lead = len(line) - len(body)
        if lead % len(INDENT):
            raise DagError(f"line {n}: indent of {lead} is not a multiple of {len(INDENT)}")
        depth = lead // len(INDENT)
        if (t := TAGS.fullmatch(body)) is not None:
            if depth != last[1] + 1:
                raise DagError(f"line {n}: a tags line sits one level under its item")
            tags = [s.strip() for s in (t.group(1) or "").split(",") if s.strip()]
            _merge(last[0], vs[last[0]], None, None, tags)
            continue
        if not body.startswith("- ") or depth >= len(parents):
            raise DagError(f"line {n}: an item is `- [handle] ...` at most one level deeper")
        parent = parents[depth]
        if (r := RELATION.fullmatch(body[2:])) is not None:
            who = MENTION.findall(r.group(1)) + MENTION.findall(r.group(2) or "")
            vs[parent].relations.append(Relation(list(dict.fromkeys(who)), r.group(2) or ""))
            del parents[depth + 1 :]
            last = (None, -2)
            continue
        if (m := HEAD.fullmatch(body[2:])) is None:
            raise DagError(f"line {n}: an item is `- [handle] [a,b,c,d] text`")
        handle, itext, box = _parts(m, coords, n)
        if handle == root:
            raise DagError(f"line {n}: the root [{root}] cannot sit under another vertex")
        v = vs.setdefault(handle, Vertex())
        _merge(handle, v, itext, box, None)
        if handle in vs[parent].children:
            raise DagError(f"line {n}: [{handle}] is under [{parent}] twice")
        vs[parent].children.append(handle)
        del parents[depth + 1 :]
        parents.append(handle)
        last = (handle, depth)
    return check(Dag(root, vs))
