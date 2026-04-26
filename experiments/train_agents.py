"""Unified training script for HCAPO, MiRA, KLong, and MemexRL agents.

Runs real training episodes and produces trained models + evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ClinicalRecruitmentEnv
from models import Action
from training.neural_policy import ACTION_SPACE, extract_state_features

# Import agents
from research.methods.hcapo_agent import HCAPOAgent
from research.methods.mira_agent import MiRAAgent
from research.methods.klong_agent import KLongAgent
from research.methods.memex_agent import MemexRLAgent


@dataclass
class TrainingConfig:
    """Training configuration."""

    agent_type: str  # hcapo, mira, klong, memex
    task_ids: List[str]
    episodes: int = 50
    eval_every: int = 10
    save_dir: Path = Path("data/trained_agents")
    results_dir: Path = Path("data/training_results")


def create_agent(agent_type: str):
    """Create agent of specified type."""
    if agent_type == "hcapo":
        return HCAPOAgent()
    elif agent_type == "mira":
        return MiRAAgent()
    elif agent_type == "klong":
        return KLongAgent()
    elif agent_type == "memex":
        return MemexRLAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode(
    env: ClinicalRecruitmentEnv,
    agent,
    task_id: str,
    training: bool = True,
) -> Dict[str, Any]:
    """Run a single episode."""
    result = env.reset(task_id)
    obs = result.observation.model_dump()

    # Reset agent state
    if hasattr(agent, "reset"):
        agent.reset()
    if hasattr(agent, "reset_milestones"):
        agent.reset_milestones()

    trajectory = []
    step = 0
    total_reward = 0.0

    while not result.done:
        # Select action
        if hasattr(agent, "select_action"):
            if isinstance(agent, MemexRLAgent):
                action_idx, info = agent.select_action(obs, step)
            else:
                action_idx, info = agent.select_action(obs)
        else:
            state = extract_state_features(obs)
            action_idx = agent.policy.select_action(state)
            info = {}

        action_type = ACTION_SPACE[action_idx]

        # Build action
        action_kwargs = {"action_type": action_type}

        def recommended_phase(current_obs: Dict[str, Any]) -> str:
            funnel = current_obs.get("current_funnel", {})
            target = max(1, int(current_obs.get("target_enrollment", 1)))
            enrolled = int(current_obs.get("enrolled_so_far", 0))
            progress = enrolled / target
            constraints = current_obs.get("active_constraints", {})
            if constraints.get("regulatory_hold_days", 0) or constraints.get("site_bottleneck", False):
                return "recovery"
            if current_obs.get("allocation_candidates"):
                return "allocation"
            if funnel.get("eligible", 0) > funnel.get("consented", 0) or current_obs.get("recontact_candidates"):
                return "conversion"
            if progress >= 0.7 or funnel.get("dropped", 0) > max(2, enrolled // 5):
                return "retention"
            return "screening"

        # Add patient_id for patient-specific actions
        if action_type == "screen_patient":
            patients = obs.get("available_patients", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")
        elif action_type == "recontact":
            patients = obs.get("recontact_candidates", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")
        elif action_type == "allocate_to_site":
            patients = obs.get("allocation_candidates", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")

        # Add site_id for allocation
        if action_type == "allocate_to_site":
            sites = obs.get("site_performance", {})
            if sites:
                action_kwargs["site_id"] = list(sites.keys())[0]

        # Add strategy for adjust_strategy
        if action_type == "adjust_strategy":
            action_kwargs["strategy_change"] = "increase_outreach"

        # Add phase for planning
        if action_type == "plan_next_phase":
            action_kwargs["target_phase"] = recommended_phase(obs)

        action = Action(**action_kwargs)
        result = env.step(action)

        reward = result.reward
        total_reward += reward

        # Record trajectory
        trajectory.append({
            "obs": obs,
            "action": action_idx,
            "action_type": action_type,
            "reward": reward,
            "done": result.done,
            "enrolled": obs.get("enrolled_so_far", 0),
            "budget_remaining": obs.get("budget_remaining", 0),
            "milestone_potential": obs.get("milestone_potential", 0),
            "info": info,
        })

        # MemexRL specific: process step
        if isinstance(agent, MemexRLAgent):
            agent.step(obs, action_idx, reward, step)

        obs = result.observation.model_dump()
        step += 1

    # Get final score
    final_score = result.info.get("final_score", 0.0)

    # Training update
    update_stats = {}
    if training and hasattr(agent, "update_from_episode"):
        update_stats = agent.update_from_episode(trajectory)

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": total_reward,
        "final_score": final_score,
        "enrolled": obs.get("enrolled_so_far", 0),
        "budget_remaining": obs.get("budget_remaining", 0),
        "update_stats": update_stats,
        "trajectory_length": len(trajectory),
    }


def train_agent(config: TrainingConfig) -> Dict[str, Any]:
    """Train an agent with the given configuration."""
    print(f"Training {config.agent_type} agent...")
    print(f"Tasks: {config.task_ids}")
    print(f"Episodes: {config.episodes}")

    agent = create_agent(config.agent_type)
    env = ClinicalRecruitmentEnv()

    # Training history
    history = []
    eval_results = []

    for episode in range(config.episodes):
        # Cycle through tasks
        task_id = config.task_ids[episode % len(config.task_ids)]

        # Training episode
        result = run_episode(env, agent, task_id, training=True)
        history.append({
            "episode": episode,
            "task_id": task_id,
            "total_reward": result["total_reward"],
            "final_score": result["final_score"],
            "enrolled": result["enrolled"],
            "steps": result["steps"],
        })

        # Evaluation
        if (episode + 1) % config.eval_every == 0:
            eval_scores = []
            for eval_task in config.task_ids:
                eval_result = run_episode(env, agent, eval_task, training=False)
                eval_scores.append(eval_result["final_score"])

            avg_score = sum(eval_scores) / len(eval_scores)
            eval_results.append({
                "episode": episode,
                "avg_score": avg_score,
                "scores": eval_scores,
            })

            print(f"Episode {episode + 1}/{config.episodes}: "
                  f"Train Score={result['final_score']:.4f}, "
                  f"Eval Avg={avg_score:.4f}")

    # Save trained agent
    config.save_dir.mkdir(parents=True, exist_ok=True)
    agent_path = config.save_dir / f"{config.agent_type}_agent.json"
    agent.save(str(agent_path))
    print(f"Saved agent to {agent_path}")

    # Save training history
    config.results_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.results_dir / f"{config.agent_type}_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "task_id", "total_reward", "final_score", "enrolled", "steps"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved history to {history_path}")

    # Save eval results
    eval_path = config.results_dir / f"{config.agent_type}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval results to {eval_path}")

    # Final evaluation
    print("\nFinal Evaluation:")
    final_scores = {}
    for task_id in config.task_ids:
        scores = []
        for _ in range(3):  # 3 eval runs per task
            result = run_episode(env, agent, task_id, training=False)
            scores.append(result["final_score"])
        avg = sum(scores) / len(scores)
        final_scores[task_id] = avg
        print(f"  {task_id}: {avg:.4f}")

    overall_avg = sum(final_scores.values()) / len(final_scores)
    print(f"  Overall Average: {overall_avg:.4f}")

    return {
        "agent_type": config.agent_type,
        "episodes": config.episodes,
        "final_scores": final_scores,
        "overall_avg": overall_avg,
        "agent_path": str(agent_path),
        "history_path": str(history_path),
    }


def train_all_agents(
    task_ids: List[str],
    episodes: int = 50,
) -> Dict[str, Any]:
    """Train all agent types and compare."""
    results = {}

    for agent_type in ["hcapo", "mira", "klong", "memex"]:
        config = TrainingConfig(
            agent_type=agent_type,
            task_ids=task_ids,
            episodes=episodes,
        )
        try:
            result = train_agent(config)
            results[agent_type] = result
        except Exception as e:
            print(f"Error training {agent_type}: {e}")
            import traceback
            traceback.print_exc()
            results[agent_type] = {"error": str(e)}

    # Summary comparison
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for agent_type, result in results.items():
        if "error" in result:
            print(f"{agent_type}: ERROR - {result['error']}")
        else:
            print(f"{agent_type}: Overall Avg = {result['overall_avg']:.4f}")

    # Save comparison
    comparison_path = Path("data/training_results/agent_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved comparison to {comparison_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train long-horizon RL agents")
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["hcapo", "mira", "klong", "memex", "all"],
        help="Agent type to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["easy_bench", "medium_bench", "hard_bench"],
        help="Task IDs to train on",
    )

    args = parser.parse_args()

    if args.agent == "all":
        train_all_agents(args.tasks, args.episodes)
    else:
        config = TrainingConfig(
            agent_type=args.agent,
            task_ids=args.tasks,
            episodes=args.episodes,
        )
        train_agent(config)


if __name__ == "__main__":
    main()
