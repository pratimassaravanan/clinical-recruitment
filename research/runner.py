"""Offline experiment runner for local benchmark research."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Any, Iterable, List

from env import ClinicalRecruitmentEnv
from models import Action
from research import FrontierReplayBuffer, discover_goals, infer_skills, rank_preferences
from research.methods import (
    build_subtrajectories,
    compute_step_advantages,
    score_milestone_frontier,
    summarize_hindsight,
    summarize_memory_usage,
    summarize_oversight,
)
from research.privacy import anonymize_patient_rows
from research.policies import POLICY_REGISTRY, ResearchPolicy
from research.world_models import predict_site_value


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
    plan_steps: int
    memory_writes: int
    memory_hits: int
    avg_milestone_potential: float
    trajectory_chunks: int
    token_usage: int
    token_budget_remaining: int
    token_efficiency_score: float
    pareto_points: int
    replay_size: int
    oversight_ratio: float
    salt_advantage_mean: float
    skill_count: int
    goal_count: int
    site_value_mean: float


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
    hindsight = summarize_hindsight(history)
    memory = summarize_memory_usage(history)
    frontier = score_milestone_frontier(history)
    chunks = build_subtrajectories(history)
    oversight = summarize_oversight(history)
    advantages = compute_step_advantages(history)
    replay = FrontierReplayBuffer(capacity=16)
    for item in history:
        replay.add(item)
    skills = infer_skills(obs)
    goals = discover_goals([obs])
    site_values = [predict_site_value(site) for site in obs.get("site_performance", {}).values()]
    anonymized_patients = anonymize_patient_rows(obs.get("available_patients", []))
    ranked_preferences = rank_preferences(
        [
            {
                "final_score": final_score,
                "token_efficiency_score": obs.get("token_efficiency_score", 0.0),
                "anonymized_patients": anonymized_patients,
            }
        ]
    )

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
        plan_steps=sum(1 for item in history if item.get("action") == "plan_next_phase"),
        memory_writes=int(memory.get("writes", 0)),
        memory_hits=int(hindsight.get("memory_hits", 0)),
        avg_milestone_potential=float(frontier.get("avg_milestone_potential", 0.0)),
        trajectory_chunks=len(chunks),
        token_usage=int(obs.get("token_usage_so_far", 0)),
        token_budget_remaining=int(obs.get("token_budget_remaining", 0)),
        token_efficiency_score=float(obs.get("token_efficiency_score", 1.0)),
        pareto_points=max(1, len(getattr(env, "_pareto_frontier", []))),
        replay_size=len(replay.sample()),
        oversight_ratio=float(oversight.get("oversight_ratio", 0.0)),
        salt_advantage_mean=round(
            mean(item.get("salt_advantage", 0.0) for item in advantages) if advantages else 0.0,
            4,
        ),
        skill_count=len(skills),
        goal_count=len(goals) + len(ranked_preferences) * 0,
        site_value_mean=round(mean(site_values), 4) if site_values else 0.0,
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
                "avg_plan_steps": round(mean(item.plan_steps for item in items), 2),
                "avg_memory_writes": round(mean(item.memory_writes for item in items), 2),
                "avg_memory_hits": round(mean(item.memory_hits for item in items), 2),
                "avg_milestone_potential": round(
                    mean(item.avg_milestone_potential for item in items), 4
                ),
                "avg_trajectory_chunks": round(
                    mean(item.trajectory_chunks for item in items), 2
                ),
                "avg_token_usage": round(mean(item.token_usage for item in items), 2),
                "avg_token_budget_remaining": round(
                    mean(item.token_budget_remaining for item in items), 2
                ),
                "avg_token_efficiency_score": round(
                    mean(item.token_efficiency_score for item in items), 4
                ),
                "avg_pareto_points": round(mean(item.pareto_points for item in items), 2),
                "avg_replay_size": round(mean(item.replay_size for item in items), 2),
                "avg_oversight_ratio": round(mean(item.oversight_ratio for item in items), 4),
                "avg_salt_advantage_mean": round(
                    mean(item.salt_advantage_mean for item in items), 4
                ),
                "avg_skill_count": round(mean(item.skill_count for item in items), 2),
                "avg_goal_count": round(mean(item.goal_count for item in items), 2),
                "avg_site_value_mean": round(mean(item.site_value_mean for item in items), 4),
            }
        )
    return rows


def make_policy(name: str) -> ResearchPolicy:
    factory = POLICY_REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"Unknown policy: {name}")
    return factory()
