"""Automated goal discovery helpers from episode metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def discover_goals(rows: Iterable[Dict[str, Any]]) -> List[str]:
    goals = []
    seen_high_dropout = False
    seen_capacity_pressure = False
    for row in rows:
        if float(row.get("avg_dropped", 0.0)) > 10:
            seen_high_dropout = True
        if float(row.get("avg_delayed_effects_triggered", 0.0)) > 120:
            seen_capacity_pressure = True
    if seen_high_dropout:
        goals.append("improve_retention")
    if seen_capacity_pressure:
        goals.append("stabilize_operations")
    if not goals:
        goals.append("increase_enrollment")
    return goals
