"""Minimal KLong-style trajectory splitting helpers for offline research."""

from __future__ import annotations

from typing import Any, Dict, List


def split_trajectory(
    history: List[Dict[str, Any]],
    window: int = 24,
    overlap: int = 8,
) -> List[Dict[str, Any]]:
    """Split a trajectory into overlapping windows with pinned boundary context."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must be >= 0 and < window")

    chunks: List[Dict[str, Any]] = []
    if not history:
        return chunks

    stride = window - overlap
    for start in range(0, len(history), stride):
        end = min(len(history), start + window)
        segment = history[start:end]
        if not segment:
            continue
        chunks.append(
            {
                "start_step": int(segment[0].get("step", start)),
                "end_step": int(segment[-1].get("step", end - 1)),
                "window": window,
                "overlap": overlap,
                "boundary_context": {
                    "first_action": segment[0].get("action"),
                    "last_action": segment[-1].get("action"),
                    "active_milestone": segment[-1].get("active_milestone", ""),
                },
                "events": segment,
            }
        )
        if end >= len(history):
            break
    return chunks
