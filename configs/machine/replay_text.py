# The GBC machine on pseudo engines replaying the recorded text-seeded PoC arm.
# gbc run configs/machine/replay_text.py

_TRACE = "runs/traces/poc30/text"

engines = {
    "vlm": {"entry": "replay", "trace": _TRACE, "capacity": 32},
    "detect": {"entry": "replay", "trace": _TRACE, "capacity": 2},
    "embed": {"entry": "replay", "trace": _TRACE},
    "cpu": {"entry": "python", "workers": 8},
}
machine = {"vlm_waiting": 32, "report": "runs/machine/replay_text/report"}
source = {"kind": "trace", "path": _TRACE}
sink = {"kind": "graphs", "path": "runs/machine/replay_text/graphs.jsonl"}
