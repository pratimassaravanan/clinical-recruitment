"""Multi-agent oversight scaffolds for offline review."""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_oversight(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    risky_steps = sum(1 for item in history if item.get("dropout") or item.get("error"))
    recovery_steps = sum(1 for item in history if item.get("action") == "adjust_strategy")
    return {
        "risky_steps": risky_steps,
        "recovery_steps": recovery_steps,
        "oversight_ratio": round(recovery_steps / max(1, risky_steps), 3),
    }
