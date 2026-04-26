"""HCAPO-oriented hindsight scaffolds for offline analysis."""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_hindsight(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a simple hindsight summary from a trajectory history."""
    followthrough = sum(1 for item in history if item.get("plan_followthrough"))
    memory_hits = sum(1 for item in history if item.get("memory_hit"))
    potential_delta = sum(float(item.get("milestone_potential_delta", 0.0)) for item in history)
    return {
        "followthrough_steps": followthrough,
        "memory_hits": memory_hits,
        "milestone_potential_gain": round(potential_delta, 4),
    }
