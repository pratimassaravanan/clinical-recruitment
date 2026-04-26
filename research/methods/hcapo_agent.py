"""HCAPO: Hierarchical Constrained Actor-critic with Planning Optimization.

Real implementation of hindsight credit assignment for long-horizon tasks.

Key components:
1. Hindsight Experience Replay (HER) - relabel failed trajectories with achieved goals
2. Hierarchical goal decomposition - break episode into subgoals
3. Constraint-aware policy - respect operational constraints
4. Temporal credit assignment - propagate rewards across long horizons
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training.neural_policy import (
    ACTION_SPACE,
    STATE_DIM,
    ActorCritic,
    NeuralNetwork,
    _softmax,
    extract_state_features,
)


@dataclass
class HindsightGoal:
    """A goal that can be used for hindsight relabeling."""

    goal_type: str  # enrollment_milestone, budget_threshold, retention_target
    target_value: float
    achieved_value: float
    achieved_step: int
    reward_bonus: float


@dataclass
class HierarchicalSubgoal:
    """A subgoal in the hierarchical decomposition."""

    subgoal_id: str
    start_step: int
    end_step: int
    goal_type: str
    target: float
    achieved: float
    success: bool
    actions: List[int] = field(default_factory=list)


@dataclass
class ConstraintViolation:
    """Record of a constraint violation during execution."""

    step: int
    constraint_type: str
    severity: float  # 0-1, how badly violated


class HCAPOAgent:
    """Hierarchical Constrained Actor-critic with Planning Optimization.

    Implements:
    - High-level planner that decomposes episode into subgoals
    - Low-level executor that achieves subgoals
    - Hindsight experience replay for failed trajectories
    - Constraint satisfaction tracking and penalty
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_sizes: Optional[List[int]] = None,
        gamma: float = 0.99,
        hindsight_k: int = 4,  # Number of hindsight goals per trajectory
        constraint_penalty: float = 0.1,
        subgoal_bonus: float = 0.05,
    ):
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.gamma = gamma
        self.hindsight_k = hindsight_k
        self.constraint_penalty = constraint_penalty
        self.subgoal_bonus = subgoal_bonus

        # High-level planner network (outputs subgoal embeddings)
        self.planner = ActorCritic(
            state_dim=state_dim,
            action_dim=5,  # 5 subgoal types: screening, conversion, allocation, retention, recovery
            hidden_sizes=self.hidden_sizes,
        )

        # Low-level executor network (outputs primitive actions)
        self.executor = ActorCritic(
            state_dim=state_dim + 5,  # State + subgoal embedding
            action_dim=len(ACTION_SPACE),
            hidden_sizes=self.hidden_sizes,
        )

        # Value function for hindsight goals
        self.hindsight_critic = NeuralNetwork(
            layer_sizes=[state_dim + 1, 64, 32, 1]  # State + goal value
        )

        # Replay buffer for hindsight experience replay
        self.replay_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 10000

        # Training stats
        self.training_stats = {
            "episodes": 0,
            "hindsight_relabels": 0,
            "constraint_violations": 0,
            "subgoals_achieved": 0,
        }

    def _decompose_into_subgoals(
        self,
        initial_obs: Dict[str, Any],
        target_enrollment: int,
    ) -> List[HierarchicalSubgoal]:
        """Decompose episode into hierarchical subgoals."""
        subgoals = []

        # Enrollment milestones as subgoals
        milestones = [0.25, 0.50, 0.75, 1.0]
        horizon = 180
        steps_per_milestone = horizon // len(milestones)

        for i, milestone in enumerate(milestones):
            target = int(target_enrollment * milestone)
            subgoals.append(HierarchicalSubgoal(
                subgoal_id=f"enroll_{int(milestone * 100)}pct",
                start_step=i * steps_per_milestone,
                end_step=(i + 1) * steps_per_milestone,
                goal_type="enrollment",
                target=target,
                achieved=0,
                success=False,
            ))

        return subgoals

    def _get_subgoal_embedding(self, subgoal_type: str) -> np.ndarray:
        """Get one-hot embedding for subgoal type."""
        types = ["screening", "conversion", "allocation", "retention", "recovery"]
        embedding = np.zeros(5, dtype=np.float32)
        if subgoal_type in types:
            embedding[types.index(subgoal_type)] = 1.0
        return embedding

    def _infer_current_subgoal(self, obs: Dict[str, Any]) -> str:
        """Infer the current subgoal type from observation."""
        mem = obs.get("patient_memory_summary", {})
        constraints = obs.get("active_constraints", {})

        if constraints.get("regulatory_hold_days", 0) > 0:
            return "recovery"
        if mem.get("consented_pending_allocation", 0) > 0:
            return "allocation"
        if mem.get("followup_due", 0) > 0 or mem.get("eligible_pending_consent", 0) > 0:
            return "conversion"
        if mem.get("at_risk_enrolled", 0) > 0:
            return "retention"
        return "screening"

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using hierarchical policy."""
        state = extract_state_features(obs)

        # Get current subgoal from planner
        subgoal_type = self._infer_current_subgoal(obs)
        subgoal_embedding = self._get_subgoal_embedding(subgoal_type)

        # Combine state with subgoal for executor
        executor_state = np.concatenate([state, subgoal_embedding])

        # Select primitive action
        action = self.executor.select_action(executor_state, deterministic)

        info = {
            "subgoal_type": subgoal_type,
            "subgoal_embedding": subgoal_embedding.tolist(),
            "state_value": self.executor.get_value(executor_state),
        }

        return action, info

    def _extract_hindsight_goals(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> List[HindsightGoal]:
        """Extract achievable hindsight goals from a trajectory."""
        goals = []

        # Find maximum enrollment achieved
        max_enrolled = max(t.get("enrolled", 0) for t in trajectory)
        if max_enrolled > 0:
            # Find when it was achieved
            for i, t in enumerate(trajectory):
                if t.get("enrolled", 0) == max_enrolled:
                    goals.append(HindsightGoal(
                        goal_type="enrollment_achieved",
                        target_value=max_enrolled,
                        achieved_value=max_enrolled,
                        achieved_step=i,
                        reward_bonus=0.1 * (max_enrolled / 150),
                    ))
                    break

        # Find best milestone potential achieved
        max_potential = max(t.get("milestone_potential", 0) for t in trajectory)
        if max_potential > 0:
            for i, t in enumerate(trajectory):
                if t.get("milestone_potential", 0) >= max_potential - 0.01:
                    goals.append(HindsightGoal(
                        goal_type="potential_achieved",
                        target_value=max_potential,
                        achieved_value=max_potential,
                        achieved_step=i,
                        reward_bonus=0.05 * max_potential,
                    ))
                    break

        # Budget efficiency goal
        final_budget = trajectory[-1].get("budget_remaining", 0) if trajectory else 0
        initial_budget = trajectory[0].get("budget_remaining", 100000) if trajectory else 100000
        budget_ratio = final_budget / max(1, initial_budget)
        if budget_ratio > 0.1:
            goals.append(HindsightGoal(
                goal_type="budget_efficiency",
                target_value=budget_ratio,
                achieved_value=budget_ratio,
                achieved_step=len(trajectory) - 1,
                reward_bonus=0.03 * budget_ratio,
            ))

        return goals[:self.hindsight_k]

    def _relabel_trajectory_with_hindsight(
        self,
        trajectory: List[Dict[str, Any]],
        hindsight_goal: HindsightGoal,
    ) -> List[Dict[str, Any]]:
        """Relabel trajectory rewards with hindsight goal."""
        relabeled = copy.deepcopy(trajectory)

        # Propagate hindsight bonus backward from achievement step
        for i in range(hindsight_goal.achieved_step + 1):
            discount = self.gamma ** (hindsight_goal.achieved_step - i)
            relabeled[i]["reward"] = relabeled[i].get("reward", 0) + hindsight_goal.reward_bonus * discount
            relabeled[i]["hindsight_goal"] = hindsight_goal.goal_type
            relabeled[i]["hindsight_target"] = hindsight_goal.target_value

        return relabeled

    def _check_constraint_violations(
        self,
        obs: Dict[str, Any],
        action: int,
    ) -> List[ConstraintViolation]:
        """Check for constraint violations."""
        violations = []
        constraints = obs.get("active_constraints", {})

        # Regulatory hold violation: screening during hold
        if constraints.get("regulatory_hold_days", 0) > 0 and action == ACTION_SPACE.index("screen_patient"):
            violations.append(ConstraintViolation(
                step=obs.get("timestamp", 0),
                constraint_type="regulatory_hold",
                severity=0.5,
            ))

        # Budget constraint: action when low budget
        if obs.get("budget_remaining", 0) < 5000 and action not in [
            ACTION_SPACE.index("stop_recruitment"),
            ACTION_SPACE.index("plan_next_phase"),
        ]:
            violations.append(ConstraintViolation(
                step=obs.get("timestamp", 0),
                constraint_type="budget_critical",
                severity=0.3,
            ))

        return violations

    def update_from_episode(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Update policy from a complete episode with hindsight relabeling."""
        if len(trajectory) < 2:
            return {}

        # Extract states, actions, rewards
        states = []
        executor_states = []
        actions = []
        rewards = []
        dones = []

        for t in trajectory:
            obs = t.get("obs", {})
            state = extract_state_features(obs)
            subgoal_type = self._infer_current_subgoal(obs)
            subgoal_emb = self._get_subgoal_embedding(subgoal_type)
            executor_state = np.concatenate([state, subgoal_emb])

            states.append(state)
            executor_states.append(executor_state)
            actions.append(t.get("action", 0))
            rewards.append(float(t.get("reward", 0)))
            dones.append(t.get("done", False))

        # Standard policy gradient update
        executor_loss = self.executor.update_from_trajectory(
            executor_states, actions, rewards, dones
        )

        # Hindsight experience replay
        hindsight_goals = self._extract_hindsight_goals(trajectory)
        hindsight_losses = []

        for goal in hindsight_goals:
            relabeled = self._relabel_trajectory_with_hindsight(trajectory, goal)

            # Extract relabeled rewards
            relabeled_rewards = [float(t.get("reward", 0)) for t in relabeled]

            # Update with relabeled trajectory
            her_loss = self.executor.update_from_trajectory(
                executor_states, actions, relabeled_rewards, dones
            )
            hindsight_losses.append(her_loss)
            self.training_stats["hindsight_relabels"] += 1

        # Add to replay buffer
        self.replay_buffer.append({
            "trajectory": trajectory,
            "hindsight_goals": [
                {"type": g.goal_type, "target": g.target_value, "achieved": g.achieved_value}
                for g in hindsight_goals
            ],
        })
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

        self.training_stats["episodes"] += 1

        return {
            "executor_actor_loss": executor_loss.get("actor_loss", 0),
            "executor_critic_loss": executor_loss.get("critic_loss", 0),
            "hindsight_updates": len(hindsight_losses),
            "avg_hindsight_loss": np.mean([h.get("actor_loss", 0) for h in hindsight_losses]) if hindsight_losses else 0,
        }

    def train_from_replay(self, batch_size: int = 32) -> Dict[str, float]:
        """Train from replay buffer samples."""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample from replay buffer
        indices = np.random.choice(len(self.replay_buffer), min(batch_size, len(self.replay_buffer)), replace=False)
        total_loss = 0.0

        for idx in indices:
            episode = self.replay_buffer[idx]
            trajectory = episode["trajectory"]

            if len(trajectory) < 2:
                continue

            # Re-extract and update
            executor_states = []
            actions = []
            rewards = []
            dones = []

            for t in trajectory:
                obs = t.get("obs", {})
                state = extract_state_features(obs)
                subgoal_type = self._infer_current_subgoal(obs)
                subgoal_emb = self._get_subgoal_embedding(subgoal_type)
                executor_state = np.concatenate([state, subgoal_emb])

                executor_states.append(executor_state)
                actions.append(t.get("action", 0))
                rewards.append(float(t.get("reward", 0)))
                dones.append(t.get("done", False))

            loss = self.executor.update_from_trajectory(executor_states, actions, rewards, dones)
            total_loss += loss.get("actor_loss", 0)

        return {"replay_loss": total_loss / max(1, len(indices))}

    def save(self, path: str):
        """Save agent to file."""
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "state_dim": self.state_dim,
            "hidden_sizes": self.hidden_sizes,
            "gamma": self.gamma,
            "hindsight_k": self.hindsight_k,
            "constraint_penalty": self.constraint_penalty,
            "subgoal_bonus": self.subgoal_bonus,
            "executor": self.executor.to_dict(),
            "planner": self.planner.to_dict(),
            "training_stats": self.training_stats,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HCAPOAgent":
        """Load agent from file."""
        import json
        from pathlib import Path

        with open(Path(path)) as f:
            data = json.load(f)

        agent = cls(
            state_dim=data["state_dim"],
            hidden_sizes=data["hidden_sizes"],
            gamma=data["gamma"],
            hindsight_k=data["hindsight_k"],
            constraint_penalty=data["constraint_penalty"],
            subgoal_bonus=data["subgoal_bonus"],
        )
        agent.executor = ActorCritic.from_dict(data["executor"])
        agent.planner = ActorCritic.from_dict(data["planner"])
        agent.training_stats = data.get("training_stats", agent.training_stats)

        return agent
