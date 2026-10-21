"""Reply parsers: a VLM reply into descriptions and named entities.

Ports of ml-gbc's ``parse_mllm_output`` and the PoC's JSON layer (``gbcpoc.vlm.json_format``):
a JSON reply is rewritten into the 2024 text layout and read by the 2024 parsers, so every rule
downstream of the reply is ml-gbc's. A result is ``{"descs": [[text, label, [ids]]],
"entities": [[entity_id, label, text]], "raw": reply}``; entities are the phrases a
description names (ml-gbc keeps only those with a position in the descriptions).
"""

import json
import re

from kohakugbc.nodes.text import ref_count, singular

MULTIPLICITIES = ("single", "multiple")
TAGGED = r"\[([^\]]+)\]\s*\[([^\]]+)\]"


def result(descs=None, entities=None, raw=None) -> dict:
    return {"descs": descs or [], "entities": entities or [], "raw": raw}


def load_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _items(obj: dict, key: str) -> list[tuple[str, str]]:
    items = []
    for item in obj.get(key) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        mult = str(item.get("multiplicity", "single")).strip().lower()
        if name:
            items.append((name, mult if mult in MULTIPLICITIES else "single"))
    return items


# ---- JSON replies to the 2024 layout (gbcpoc.vlm.json_format) ---------------------------------


def legacy_image(text: str) -> str:
    obj = load_json(text)
    if obj is None:
        return ""
    elements = _items(obj, "elements")
    mult_of = {name.lower(): mult for name, mult in elements}

    def tag(match):
        name = match.group(1)
        mult = mult_of.get(name.lower())
        return f"[{name}][{mult}]" if mult else name

    concise = re.sub(r"\[([^\[\]]+)\](?!\s*\[)", tag, str(obj.get("concise_caption", "")))
    lines = [
        f"Detailed Caption: {obj.get('detailed_caption', '')}",
        "",
        "Top-Level Element Identification:",
    ]
    lines += [f"- [{name}][{mult}]" for name, mult in elements]
    lines += ["", f"Concise Formatted Caption: {concise}"]
    return "\n".join(lines)


def legacy_entity(text: str) -> str:
    obj = load_json(text)
    if obj is None:
        return ""
    if not obj.get("present"):
        return (
            "Object Present: No\nDetailed Caption: N/A\nProminent Features: No\n"
            "Identification of Prominent Features: N/A"
        )
    features = _items(obj, "features")
    lines = ["Object Present: Yes", f"Detailed Caption: {obj.get('detailed_caption', '')}"]
    if features:
        lines += ["Prominent Features: Yes", "Identification of Prominent Features:"]
        lines += [f"- {name}: [{mult}]" for name, mult in features]
    else:
        lines += ["Prominent Features: No", "Identification of Prominent Features: N/A"]
    return "\n".join(lines)


def legacy_relation(text: str) -> str:
    obj = load_json(text)
    if obj is None:
        return ""
    return "\n".join(f"- {s}" for s in obj.get("relations") or [] if isinstance(s, str))


def legacy_composition(text: str) -> str:
    obj = load_json(text)
    if obj is None:
        return ""
    lines = [f"Composition: {obj.get('composition', '')}", "General descriptions:"]
    lines += [f"- {s}" for s in obj.get("general_descriptions") or [] if isinstance(s, str)]
    return "\n".join(lines)


# ---- entity names (parse_mllm_output._transform_text_label_to_entity_info) --------------------


def name_entity(text: str, label: str, node_id: str, seen: set) -> list | None:
    """``[entity_id, label, text]`` of a tagged phrase, or None for an empty or repeated one."""
    text = text.strip().lower()
    # Numbered and left/right names are instances of one object
    if " " in text:
        parts = text.split(" ")
        if parts[-1].isdigit():
            text = " ".join(parts[:-1])
            label = "multiple"
        if len(parts) > 2 and " ".join(parts[-2:]) in ("on left", "on right"):
            text = " ".join(parts[:-2])
            label = "multiple"
    if text in seen or text.strip() == "":
        return None
    seen.add(text)
    entity_id = f"{node_id}_{text}" if node_id else text
    return [entity_id, label if label in MULTIPLICITIES else "entity", text]


def concise(caption: str, node_id: str = "", seen: set | None = None) -> dict:
    """A concise caption with tagged phrases (``concise_caption_to_query_result``)."""
    seen = set() if seen is None else seen
    desc = [re.sub(TAGGED, r"\1", caption), "short", [node_id]]
    entities = []
    for text, label in re.findall(TAGGED, caption):
        entity = name_entity(text, label, node_id, seen)
        if entity is not None:
            entities.append(entity)
    return result([desc], entities)


# ---- image query -------------------------------------------------------------------------------


def image(reply: str, node_id: str = "", max_features: int = 10) -> dict:
    """``parse_image_json``: detailed and concise captions, the elements they name."""
    text = legacy_image(reply)
    detailed, concise_caption, mode = "", "", None
    seen: set = set()
    identified = []
    for line in text.split("\n"):
        normalized = line.lstrip("#").strip()
        if normalized.startswith("Detailed Caption:"):
            mode = "detailed"
            detailed += normalized[len("Detailed Caption:") :].strip() + " "
        elif normalized.startswith("Top-Level Element Identification:"):
            mode = "features"
        elif normalized.startswith("Concise Formatted Caption:"):
            mode = "concise"
            inline = normalized[len("Concise Formatted Caption:") :].strip()
            if inline:
                concise_caption += inline
                break
            mode = "concise_next_line"
        elif mode == "detailed" and line.strip():
            detailed += line.strip()
        elif mode == "concise_next_line" and line.strip():
            concise_caption += line.strip()
            break
        elif mode == "features" and line.startswith("- "):
            tagged = re.findall(TAGGED, line)
            if len(tagged) == 1 and len(seen) < max_features:
                entity = name_entity(tagged[0][0], tagged[0][1], "", seen)
                if entity is not None:
                    identified.append(entity)

    out = concise(concise_caption, node_id, seen) if concise_caption else result()
    if detailed:
        out["descs"].insert(0, [detailed.strip(), "detail", [node_id]])
    texts = [d[0] for d in out["descs"]]
    out["entities"] = [e for e in out["entities"] + identified if ref_count(texts, e[2]) > 0]
    out["raw"] = reply
    return out


# ---- entity query ------------------------------------------------------------------------------


def entity_text(text: str, max_features: int = 10):
    """``parse_entity_text``: (detailed caption, short captions, features), or None if absent."""
    if "Object Present: No" in text or "Object Presence: No" in text:
        return None
    detailed, shorts, features, seen = "", [], [], set()
    mode = None
    for line in text.split("\n"):
        normalized = line.lstrip("#").strip()
        if mode == "detailed" and not line.strip():
            mode = None
        elif normalized.startswith("Detailed Caption:"):
            mode = "detailed"
            detailed = line[len("Detailed Caption:") :].strip()
        elif normalized.startswith("Prominent Features:"):
            mode = None
        elif normalized.startswith("Short Captions:"):
            mode = "short"
        elif normalized.startswith("Identification of Prominent Features:"):
            mode = "features"
        elif line.startswith("- ") and mode == "short":
            short = line[2:].strip()
            if short and short != "N/A":
                shorts.append(short)
        elif mode == "features" and line.startswith("- "):
            if ":" not in line:
                continue
            feature, rest = line[2:].split(":", 1)
            match = re.match(r"\s*\[([^\]]+)\]", rest.strip())
            if match:
                label = match.group(1).lower()
                feature = feature.strip().lower()
                if feature in seen:
                    continue
                seen.add(feature)
                features.append((feature, label))
                if len(features) == max_features:
                    break
        elif mode == "detailed":
            detailed += " " + line.strip()
    if detailed in ("N/A", ""):
        detailed = None
    return detailed, shorts, features


def entity(reply: str, node_id: str, parent_text: str | None) -> dict:
    """``parse_entity_json``: the entity's captions and the features they name."""
    out = result(raw=reply)
    parsed = entity_text(legacy_entity(reply))
    if parsed is None:
        return out
    detailed, shorts, features = parsed
    texts = []
    if detailed:
        out["descs"].append([detailed, "detail", [node_id]])
        texts.append(detailed)
    for short in shorts:
        out["descs"].append([short, "short", [node_id]])
        texts.append(short)
    for text, label in features:
        # The entity itself is not one of its features
        if singular(text) == singular(parent_text):
            continue
        if ref_count(texts, text) > 0:
            label = label if label in MULTIPLICITIES else "entity"
            out["entities"].append([f"{node_id}_{text}", label, text])
    return out


def feature_captions(raw: str | None) -> dict[str, str]:
    """Feature name (lower case) -> its caption, from an entity reply that writes them."""
    captions = {}
    for item in (load_json(raw or "") or {}).get("features") or []:
        if isinstance(item, dict) and item.get("name") and item.get("caption"):
            captions[str(item["name"]).strip().lower()] = str(item["caption"]).strip()
    return captions


# ---- relation and composition queries ----------------------------------------------------------


def relation_sentences(text: str) -> list[str]:
    """Listed sentences, repeats dropped in first-seen order."""
    pattern = re.compile(r"(?:^|\n)-\s*([^\n]+)|(?:^|\n)\d+\.\s*([^\n]+)")
    return list(dict.fromkeys(m[0] or m[1] for m in pattern.findall(text)))


def relation(reply: str, entities: list) -> dict:
    """``parse_relation_json``: each sentence naming at least two of ``[[id, text], ...]``."""
    descs = []
    for sentence in relation_sentences(legacy_relation(reply)):
        ids, modified = [], sentence
        for entity_id, text in entities:
            pattern = re.compile(r"\b\[?" + re.escape(text) + r"\]?\b", re.IGNORECASE)
            if pattern.findall(modified):
                ids.append(entity_id)
                bracketed = re.compile(r"\[" + re.escape(text) + r"\]", re.IGNORECASE)
                modified = bracketed.sub(text, modified)
        if len(ids) >= 2:
            descs.append([modified, "relation", ids])
    return result(descs, [], reply)


def composition_text(raw: str) -> tuple[str, list[str]]:
    composition, descriptions = "", []
    reading_composition = reading_descriptions = False
    for line in raw.split("\n"):
        normalized = line.lstrip("#").strip()
        if normalized.startswith("Composition:"):
            reading_composition = True
            composition = normalized.replace("Composition:", "").strip()
            continue
        if normalized.startswith(("General descriptions:", "General Descriptions:")):
            reading_composition = False
            reading_descriptions = True
            remaining = (
                normalized.replace("General descriptions:", "")
                .replace("General Descriptions:", "")
                .strip()
            )
            if remaining:
                descriptions.append(remaining)
            continue
        if reading_composition:
            composition += " " + line.strip()
        if reading_descriptions and line.startswith("-"):
            descriptions.append(line[2:].strip())
    return composition, descriptions


def composition(reply: str, node_id: str, parts: list) -> dict:
    """``parse_composition_json``: the composition caption naming every part of
    ``[[id, label, text], ...]`` (appended when the reply leaves one out), then the general
    descriptions."""
    comp, descriptions = composition_text(legacy_composition(reply))
    for _, _, text in parts:
        if ref_count([comp], text) == 0:
            comp = text if comp == "" else comp + " " + text
    descs = [[comp, "composition", [node_id]]]
    descs += [[d, "short", [node_id]] for d in descriptions]
    return result(descs, [list(p) for p in parts], reply)
