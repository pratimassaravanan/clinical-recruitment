"""Memex-style indexed-memory helpers for offline analysis."""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_memory_usage(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize memory writes and retrieval hits from an episode history."""
    writes = sum(1 for item in history if item.get("action") == "summarize_and_index")
    retrievals = sum(
        1 for item in history if item.get("action") == "retrieve_relevant_history"
    )
    hits = sum(1 for item in history if item.get("memory_hit"))
    return {
        "writes": writes,
        "retrievals": retrievals,
        "hits": hits,
    }
