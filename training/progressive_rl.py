"""Progressive horizon training scaffolds for offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from load_traces import PROGRESSIVE_HORIZONS, PUBLIC_TASKS, make_stage_task_id
from research.runner import run_episode, make_policy


@dataclass
class ProgressiveStageResult:
    base_task: str
    task: str
    horizon_days: int
    policy: str
    final_score: float
    total_reward: float
    enrolled: int
    target: int
    budget_remaining: float
    plan_steps: int
    memory_writes: int
    memory_hits: int
    avg_milestone_potential: float
    trajectory_chunks: int
    token_usage: int
    token_budget_remaining: int
    token_efficiency_score: float


def progressive_task_sequence(base_task: str) -> List[str]:
    return [make_stage_task_id(base_task, horizon) for horizon in PROGRESSIVE_HORIZONS]


def run_progressive_sequence(policy_name: str, base_task: str) -> List[ProgressiveStageResult]:
    results: List[ProgressiveStageResult] = []
    for stage_task_id in progressive_task_sequence(base_task):
        policy = make_policy(policy_name)
        summary = run_episode(stage_task_id, policy)
        horizon_days = int(stage_task_id.rsplit("_", 1)[-1])
        results.append(
            ProgressiveStageResult(
                base_task=base_task,
                task=stage_task_id,
                horizon_days=horizon_days,
                policy=policy_name,
                final_score=summary.final_score,
                total_reward=summary.total_reward,
                enrolled=summary.enrolled,
                target=summary.target,
                budget_remaining=summary.budget_remaining,
                plan_steps=summary.plan_steps,
                memory_writes=summary.memory_writes,
                memory_hits=summary.memory_hits,
                avg_milestone_potential=summary.avg_milestone_potential,
                trajectory_chunks=summary.trajectory_chunks,
                token_usage=summary.token_usage,
                token_budget_remaining=summary.token_budget_remaining,
                token_efficiency_score=summary.token_efficiency_score,
            )
        )
    return results


def default_progressive_tasks() -> List[str]:
    task_ids: List[str] = []
    for base_task in PUBLIC_TASKS:
        task_ids.extend(progressive_task_sequence(base_task))
    return task_ids
