"""KLong-oriented wrappers around trajectory splitting utilities."""

from __future__ import annotations

from typing import Any, Dict, List

from training.trajectory_splitter import split_trajectory


def build_subtrajectories(
    history: List[Dict[str, Any]],
    window: int = 24,
    overlap: int = 8,
) -> List[Dict[str, Any]]:
    """Expose overlapping subtrajectories for offline long-horizon studies."""
    return split_trajectory(history, window=window, overlap=overlap)
