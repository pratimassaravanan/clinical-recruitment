"""Async-style training orchestration scaffold."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List

from training.train_offline_policy import train_policy


def run_async_training(task_groups: Iterable[List[str]], epochs: int = 2) -> List[Dict[str, Any]]:
    groups = list(task_groups)
    results: List[Dict[str, Any]] = []

    def _worker(task_ids: List[str]) -> Dict[str, Any]:
        _, history = train_policy(task_ids, epochs=epochs, policy_type="mlp")
        last = history[-1]
        return {
            "tasks": ",".join(task_ids),
            "avg_final_score": last.avg_final_score,
            "avg_total_reward": last.avg_total_reward,
        }

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(groups)))) as executor:
        for row in executor.map(_worker, groups):
            results.append(row)
    return results
