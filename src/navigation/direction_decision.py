"""Direction decision from free/blocked sectors, proposal Sub-bab 3.2.4.7.

Priority CTR -> L -> R -> FL -> FR; the first free sector decides:
CTR -> FORWARD, L or FL -> TURN_LEFT, R or FR -> TURN_RIGHT; all blocked -> STOP.
No path planning, mapping or motor control is involved (interface-level decision only).
"""

from __future__ import annotations

PRIORITY = ("CTR", "L", "R", "FL", "FR")
COMMAND = {"CTR": "FORWARD", "L": "TURN_LEFT", "FL": "TURN_LEFT", "R": "TURN_RIGHT", "FR": "TURN_RIGHT"}
DIRECTIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def decide_direction(sectors: list[dict], priority=PRIORITY) -> tuple[str, str | None]:
    """Return (command, chosen sector name or None for STOP)."""
    status = {s["name"]: s["blocked"] for s in sectors}
    for name in priority:
        if name in status and not status[name]:
            return COMMAND[name], name
    return "STOP", None
