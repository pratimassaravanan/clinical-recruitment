"""SALT-style step-level advantage approximation."""

from __future__ import annotations

from typing import Any, Dict, List


def compute_step_advantages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    running = 0.0
    rows: List[Dict[str, Any]] = []
    for item in reversed(history):
        running = float(item.get("reward", 0.0)) + 0.9 * running
        rows.append({
            "step": item.get("step", 0),
            "action": item.get("action", ""),
            "salt_advantage": round(running, 4),
        })
    rows.reverse()
    return rows
