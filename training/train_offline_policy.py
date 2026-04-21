"""Train a lightweight offline policy on staged benchmark tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import csv

from env import ClinicalRecruitmentEnv
from models import Action
from training.offline_policy import (
    LinearPolicy,
    MLPPolicy,
    TrainingExample,
    action_payload_for_type,
    extract_features,
)


@dataclass
class EpochResult:
    epoch: int
    avg_total_reward: float
    avg_final_score: float
    avg_token_efficiency: float


PolicyLike = LinearPolicy | MLPPolicy


def run_training_episode(task_id: str, policy: PolicyLike, epsilon: float) -> tuple[list[TrainingExample], Dict[str, float]]:
    env = ClinicalRecruitmentEnv()
    result = env.reset(task_id)
    obs = result.observation.model_dump()
    from inference import PolicyState

    policy_state = PolicyState()
    policy_state.reset(obs)
    examples: List[TrainingExample] = []
    step_num = 0
    final_score = 0.0

    while not result.done:
        features = extract_features(obs, step_num)
        action_type = policy.choose_action_type(features, epsilon=epsilon)
        action_payload = action_payload_for_type(obs, step_num, policy_state, action_type)
        result = env.step(Action(**action_payload))
        raw = result.model_dump()
        policy_state.update(obs, action_payload, raw, step_num)
        obs = raw["observation"]
        examples.append(
            TrainingExample(
                features=features,
                action_type=action_payload["action_type"],
                total_reward=float(raw.get("reward", 0.0) or 0.0),
            )
        )
        step_num += 1
        if result.done:
            final_score = float(raw.get("info", {}).get("final_score", 0.0) or 0.0)

    metrics = {
        "total_reward": float(env.state().total_reward),
        "final_score": final_score,
        "token_efficiency": float(obs.get("token_efficiency_score", 1.0) or 0.0),
    }
    return examples, metrics


def train_policy(
    task_ids: List[str],
    epochs: int = 8,
    learning_rate: float = 0.05,
    seed: int = 0,
    policy_type: str = "linear",
) -> tuple[PolicyLike, List[EpochResult]]:
    if policy_type == "mlp":
        policy: PolicyLike = MLPPolicy(seed=seed)
    else:
        policy = LinearPolicy(seed=seed)
    history: List[EpochResult] = []
    for epoch in range(1, epochs + 1):
        epoch_examples: List[TrainingExample] = []
        rewards: List[float] = []
        scores: List[float] = []
        token_scores: List[float] = []
        epsilon = max(0.05, 0.35 - epoch * 0.03)
        for task_id in task_ids:
            examples, metrics = run_training_episode(task_id, policy, epsilon=epsilon)
            epoch_examples.extend(examples)
            rewards.append(metrics["total_reward"])
            scores.append(metrics["final_score"])
            token_scores.append(metrics["token_efficiency"])
        policy.update(epoch_examples, learning_rate=learning_rate)
        history.append(
            EpochResult(
                epoch=epoch,
                avg_total_reward=round(sum(rewards) / max(1, len(rewards)), 4),
                avg_final_score=round(sum(scores) / max(1, len(scores)), 4),
                avg_token_efficiency=round(sum(token_scores) / max(1, len(token_scores)), 4),
            )
        )
    return policy, history


def save_training_outputs(
    output_dir: Path,
    policy: PolicyLike,
    history: List[EpochResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save(output_dir / "offline_policy_weights.json")
    rows = [asdict(item) for item in history]
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(task_ids: List[str], policy: PolicyLike) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for task_id in task_ids:
        _, metrics = run_training_episode(task_id, policy, epsilon=0.0)
        rows.append(
            {
                "task": task_id,
                "final_score": round(metrics["final_score"], 4),
                "total_reward": round(metrics["total_reward"], 4),
                "token_efficiency": round(metrics["token_efficiency"], 4),
            }
        )
    return rows


def save_evaluation_outputs(output_dir: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with (output_dir / "training_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
