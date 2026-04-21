"""Offline experiment runner for local benchmark research."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Any, Iterable, List

from env import ClinicalRecruitmentEnv
from models import Action
from research.policies import POLICY_REGISTRY, ResearchPolicy


@dataclass
class EpisodeSummary:
    task: str
    policy: str
    steps: int
    final_score: float
    total_reward: float
    enrolled: int
    target: int
    budget_remaining: float
    dropped: int
    screened: int
    milestones_hit: int
    delayed_effects_triggered: int
    strategy_steps: int
    recontacts: int
    allocations: int


def run_episode(task: str, policy: ResearchPolicy) -> EpisodeSummary:
    env = ClinicalRecruitmentEnv()
    result = env.reset(task)
    obs = result.observation.model_dump()
    policy.reset(obs)

    steps = 0
    final_score = 0.0

    while not result.done:
        action_payload = policy.act(obs, steps)
        result = env.step(Action(**action_payload))
        raw = result.model_dump()
        policy.update(obs, action_payload, raw, steps)
        obs = raw["observation"]
        steps += 1
        if result.done:
            final_score = float(raw.get("info", {}).get("final_score", 0.0) or 0.0)

    history = env.get_history()
    funnel = obs.get("current_funnel", {})
    milestones = obs.get("milestones", {})

    return EpisodeSummary(
        task=task,
        policy=policy.name,
        steps=steps,
        final_score=final_score,
        total_reward=float(env.state().total_reward),
        enrolled=int(obs.get("enrolled_so_far", 0)),
        target=int(obs.get("target_enrollment", 0)),
        budget_remaining=float(obs.get("budget_remaining", 0.0)),
        dropped=int(funnel.get("dropped", 0)),
        screened=int(funnel.get("screened", 0)),
        milestones_hit=sum(1 for reached in milestones.values() if reached),
        delayed_effects_triggered=sum(
            int(item.get("delayed_effects_triggered", 0)) for item in history
        ),
        strategy_steps=sum(1 for item in history if item.get("action") == "adjust_strategy"),
        recontacts=sum(1 for item in history if item.get("action") == "recontact"),
        allocations=sum(1 for item in history if item.get("action") == "allocate_to_site"),
    )


def aggregate_results(summaries: Iterable[EpisodeSummary]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[EpisodeSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.policy, summary.task), []).append(summary)

    rows: List[Dict[str, Any]] = []
    for (policy, task), items in sorted(grouped.items()):
        rows.append(
            {
                "policy": policy,
                "task": task,
                "episodes": len(items),
                "avg_final_score": round(mean(item.final_score for item in items), 4),
                "avg_total_reward": round(mean(item.total_reward for item in items), 4),
                "avg_enrolled": round(mean(item.enrolled for item in items), 2),
                "avg_budget_remaining": round(mean(item.budget_remaining for item in items), 2),
                "avg_dropped": round(mean(item.dropped for item in items), 2),
                "avg_screened": round(mean(item.screened for item in items), 2),
                "avg_milestones_hit": round(mean(item.milestones_hit for item in items), 2),
                "avg_delayed_effects_triggered": round(
                    mean(item.delayed_effects_triggered for item in items), 2
                ),
                "avg_strategy_steps": round(mean(item.strategy_steps for item in items), 2),
                "avg_recontacts": round(mean(item.recontacts for item in items), 2),
                "avg_allocations": round(mean(item.allocations for item in items), 2),
            }
        )
    return rows


def make_policy(name: str) -> ResearchPolicy:
    factory = POLICY_REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"Unknown policy: {name}")
    return factory()
