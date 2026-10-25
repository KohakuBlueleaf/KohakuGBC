"""Admission over every limited lane (``kohakugbc.machine.build.admission_of``), on the signals
Togeari's dispatcher hands a policy."""

from types import SimpleNamespace

import pytest

from kohakugbc.machine.build import admission_of

LANES = {n: SimpleNamespace(name=n) for n in ("vlm", "detect", "embed", "cpu")}


def signals(**backlog) -> dict:
    lanes = {n: {"queue": 0, "inflight": 0, "units_up": 1, "waiting": 0} for n in LANES}
    for name, (queue, waiting) in backlog.items():
        lanes[name] |= {"queue": queue, "waiting": waiting}
    return {"lanes": lanes}


def pool(jobs: int):
    return SimpleNamespace(jobs=jobs, bytes=0)


def test_every_limited_lane_bounds_admission():
    rule = admission_of(LANES, {"vlm_waiting": 8, "waiting": {"detect": 16}, "max_jobs": 100})
    assert rule.allows(pool(0), signals())
    # A detect backlog at its limit (its queue and its units' own waiting) stops admission
    assert not rule.allows(pool(0), signals(detect=(10, 6)))
    assert rule.allows(pool(0), signals(detect=(10, 5)))
    assert not rule.allows(pool(0), signals(vlm=(8, 0)))
    # A lane without a limit never stops it; the job cap always does
    assert rule.allows(pool(0), signals(embed=(10_000, 0), cpu=(10_000, 0)))
    assert not rule.allows(pool(100), signals())


def test_the_vlm_limit_stands_alone_without_others():
    rule = admission_of(LANES, {"vlm_waiting": 4, "max_jobs": 10})
    assert len(rule.rules) == 1
    assert rule.allows(pool(0), signals(detect=(10_000, 0)))
    assert not rule.allows(pool(0), signals(vlm=(4, 0)))


@pytest.mark.parametrize(
    "m, match",
    [
        ({"vlm_waiting": 4, "waiting": {"detector": 8}, "max_jobs": 10}, "no lane"),
        ({"vlm_waiting": 4, "waiting": {"vlm": 8}, "max_jobs": 10}, "vlm_waiting"),
        ({"vlm_waiting": 4, "waiting": {"detect": 0}, "max_jobs": 10}, "at least 1"),
        ({"vlm_waiting": 4, "max_jobs": 0}, "at least 1"),
    ],
)
def test_limits_that_cannot_hold_are_refused(m, match):
    with pytest.raises(ValueError, match=match):
        admission_of(LANES, m)
