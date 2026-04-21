"""MiRA-oriented milestone potential helpers for offline analysis."""

from __future__ import annotations

from typing import Any, Dict, List


def score_milestone_frontier(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate frontier quality signals from an episode history."""
    if not history:
        return {"avg_milestone_potential": 0.0, "frontier_switches": 0}

    potentials = [float(item.get("milestone_potential", 0.0)) for item in history]
    frontiers = [item.get("active_milestone", "") for item in history if item.get("active_milestone")]
    switches = sum(1 for idx in range(1, len(frontiers)) if frontiers[idx] != frontiers[idx - 1])
    return {
        "avg_milestone_potential": round(sum(potentials) / max(1, len(potentials)), 4),
        "frontier_switches": switches,
    }
