"""Human-in-the-loop preference scaffolds for offline ranking."""

from __future__ import annotations

from typing import Any, Dict, List


def rank_preferences(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        episodes,
        key=lambda item: (
            float(item.get("final_score", 0.0)),
            float(item.get("token_efficiency_score", 0.0)),
        ),
        reverse=True,
    )
