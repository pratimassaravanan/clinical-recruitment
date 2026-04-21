"""MiRA: Milestone-based Reward Augmentation.

Real implementation of potential-based reward shaping with learned milestone critic.

Key components:
1. Potential function learning - estimate value of reaching milestones
2. Reward shaping - augment rewards with potential difference
3. Subgoal decomposition - break task into milestone targets
4. Multi-objective balancing - handle enrollment/budget/retention tradeoffs
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
    extract_state_features,
)


@dataclass
class Milestone:
    """A milestone in the episode."""

    milestone_id: str
    threshold: float  # e.g., 0.25 for 25% enrollment
    achieved: bool = False
    achieved_step: Optional[int] = None
    reward_value: float = 0.0


@dataclass
class PotentialState:
    """State representation for potential function."""

    enrollment_progress: float  # 0-1
    budget_ratio: float  # 0-1
    time_ratio: float  # 0-1 (time remaining)
    milestone_distance: float  # Distance to next milestone
    funnel_health: float  # Overall funnel quality


class MiRACritic:
    """Learned potential function for milestone-based reward shaping."""

    def __init__(
        self,
        state_dim: int = 5,  # PotentialState features
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.001,
    ):
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes or [32, 16]

        # Potential network: maps state to scalar potential value
        self.network = NeuralNetwork(
            layer_sizes=[state_dim] + self.hidden_sizes + [1],
            learning_rate=learning_rate,
        )

        # Target network for stability
        self.target_network = NeuralNetwork(
            layer_sizes=[state_dim] + self.hidden_sizes + [1],
            learning_rate=learning_rate,
        )
        self._sync_target()

        self.update_count = 0
        self.target_update_freq = 100

    def _sync_target(self):
        """Sync target network with main network."""
        self.target_network.weights = [w.copy() for w in self.network.weights]
        self.target_network.biases = [b.copy() for b in self.network.biases]

    def _extract_potential_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract features for potential function."""
        enrolled = obs.get("enrolled_so_far", 0)
        target = obs.get("target_enrollment", 100)
        budget = obs.get("budget_remaining", 0)
        initial_budget = 150000  # Approximate
        time_remaining = obs.get("time_to_deadline_days", 180)
        max_time = 180

        enrollment_progress = enrolled / max(1, target)
        budget_ratio = budget / max(1, initial_budget)
        time_ratio = time_remaining / max_time

        # Distance to next milestone
        milestones = [0.25, 0.50, 0.75, 1.0]
        milestone_distance = 1.0
        for m in milestones:
            if enrollment_progress < m:
                milestone_distance = m - enrollment_progress
                break

        # Funnel health: weighted combination of funnel metrics
        funnel = obs.get("current_funnel", {})
        screened = funnel.get("screened", 0)
        consented = funnel.get("consented", 0)
        dropped = funnel.get("dropped", 0)
        funnel_health = (consented / max(1, screened + 1)) * (1 - dropped / max(1, enrolled + 1))
        funnel_health = max(0, min(1, funnel_health))

        return np.array([
            enrollment_progress,
            budget_ratio,
            time_ratio,
            milestone_distance,
            funnel_health,
        ], dtype=np.float32)

    def get_potential(self, obs: Dict[str, Any], use_target: bool = False) -> float:
        """Get potential value for a state."""
        features = self._extract_potential_features(obs)
        network = self.target_network if use_target else self.network
        value, _ = network.forward(features)
        return float(value[0] if value.ndim > 0 else value)

    def compute_shaped_reward(
        self,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        original_reward: float,
        gamma: float = 0.99,
    ) -> float:
        """Compute potential-based shaped reward.

        F(s, s') = gamma * Phi(s') - Phi(s)
        """
        phi_s = self.get_potential(obs)
        phi_s_next = self.get_potential(next_obs, use_target=True)

        shaping_bonus = gamma * phi_s_next - phi_s
        shaped_reward = original_reward + shaping_bonus

        return shaped_reward

    def update(
        self,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        reward: float,
        done: bool,
        gamma: float = 0.99,
    ) -> float:
        """Update potential function using TD learning."""
        features = self._extract_potential_features(obs)
        next_features = self._extract_potential_features(next_obs)

        # Current value
        current_value, activations = self.network.forward(features)
        current_value = float(current_value[0] if current_value.ndim > 0 else current_value)

        # Target value
        if done:
            target_value = reward
        else:
            next_value, _ = self.target_network.forward(next_features)
            next_value = float(next_value[0] if next_value.ndim > 0 else next_value)
            target_value = reward + gamma * next_value

        # TD error
        td_error = target_value - current_value

        # Gradient update
        grad = np.array([td_error])
        weight_grads, bias_grads = self.network.backward(activations, -grad)
        self.network.update(weight_grads, bias_grads)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._sync_target()

        return td_error ** 2


class MiRAAgent:
    """Milestone-based Reward Augmentation Agent.

    Implements:
    - Learned potential function for reward shaping
    - Milestone tracking and bonus allocation
    - Multi-objective value decomposition
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_sizes: Optional[List[int]] = None,
        gamma: float = 0.99,
        milestone_bonus: float = 0.1,
        shaping_weight: float = 0.5,
    ):
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.gamma = gamma
        self.milestone_bonus = milestone_bonus
        self.shaping_weight = shaping_weight

        # Main policy network
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=len(ACTION_SPACE),
            hidden_sizes=self.hidden_sizes,
            gamma=gamma,
        )

        # Milestone potential critic
        self.potential_critic = MiRACritic()

        # Milestone tracking
        self.milestones = self._init_milestones()
        self.milestone_history: List[Dict[str, Any]] = []

        # Training stats
        self.training_stats = {
            "episodes": 0,
            "milestones_achieved": 0,
            "total_shaped_reward": 0.0,
            "potential_updates": 0,
        }

    def _init_milestones(self) -> List[Milestone]:
        """Initialize milestone tracking."""
        return [
            Milestone("25pct", 0.25, reward_value=0.02),
            Milestone("50pct", 0.50, reward_value=0.03),
            Milestone("75pct", 0.75, reward_value=0.04),
            Milestone("100pct", 1.00, reward_value=0.06),
        ]

    def reset_milestones(self):
        """Reset milestones for new episode."""
        self.milestones = self._init_milestones()

    def _check_milestone_achievement(
        self,
        obs: Dict[str, Any],
        step: int,
    ) -> float:
        """Check and return bonus for newly achieved milestones."""
        enrolled = obs.get("enrolled_so_far", 0)
        target = obs.get("target_enrollment", 100)
        progress = enrolled / max(1, target)

        bonus = 0.0
        for milestone in self.milestones:
            if not milestone.achieved and progress >= milestone.threshold:
                milestone.achieved = True
                milestone.achieved_step = step
                bonus += milestone.reward_value * self.milestone_bonus
                self.training_stats["milestones_achieved"] += 1
                self.milestone_history.append({
                    "milestone_id": milestone.milestone_id,
                    "step": step,
                    "progress": progress,
                })

        return bonus

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using policy with milestone awareness."""
        state = extract_state_features(obs)
        action = self.policy.select_action(state, deterministic)

        info = {
            "state_value": self.policy.get_value(state),
            "potential": self.potential_critic.get_potential(obs),
            "next_milestone": self._get_next_milestone(),
        }

        return action, info

    def _get_next_milestone(self) -> Optional[str]:
        """Get the next unachieved milestone."""
        for milestone in self.milestones:
            if not milestone.achieved:
                return milestone.milestone_id
        return None

    def compute_shaped_reward(
        self,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        original_reward: float,
        step: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute full shaped reward with milestone bonus."""
        # Potential-based shaping
        shaped = self.potential_critic.compute_shaped_reward(
            obs, next_obs, original_reward, self.gamma
        )

        # Milestone achievement bonus
        milestone_bonus = self._check_milestone_achievement(next_obs, step)

        # Combine
        total_reward = (
            (1 - self.shaping_weight) * original_reward
            + self.shaping_weight * shaped
            + milestone_bonus
        )

        self.training_stats["total_shaped_reward"] += total_reward

        breakdown = {
            "original": original_reward,
            "shaped": shaped,
            "milestone_bonus": milestone_bonus,
            "total": total_reward,
        }

        return total_reward, breakdown

    def update_from_episode(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Update policy and potential critic from episode."""
        if len(trajectory) < 2:
            return {}

        # Reset milestones for potential re-evaluation
        self.reset_milestones()

        states = []
        actions = []
        shaped_rewards = []
        dones = []
        potential_losses = []

        for i in range(len(trajectory) - 1):
            t = trajectory[i]
            t_next = trajectory[i + 1]

            obs = t.get("obs", {})
            next_obs = t_next.get("obs", {})
            original_reward = float(t.get("reward", 0))
            done = t.get("done", False)

            state = extract_state_features(obs)
            action = t.get("action", 0)

            # Compute shaped reward
            shaped_reward, _ = self.compute_shaped_reward(
                obs, next_obs, original_reward, i
            )

            states.append(state)
            actions.append(action)
            shaped_rewards.append(shaped_reward)
            dones.append(done)

            # Update potential critic
            loss = self.potential_critic.update(
                obs, next_obs, original_reward, done, self.gamma
            )
            potential_losses.append(loss)
            self.training_stats["potential_updates"] += 1

        # Add final state
        final_obs = trajectory[-1].get("obs", {})
        states.append(extract_state_features(final_obs))
        dones.append(trajectory[-1].get("done", True))

        # Policy gradient update with shaped rewards
        policy_loss = self.policy.update_from_trajectory(
            states, actions, shaped_rewards, dones[:-1]
        )

        self.training_stats["episodes"] += 1

        return {
            "policy_actor_loss": policy_loss.get("actor_loss", 0),
            "policy_critic_loss": policy_loss.get("critic_loss", 0),
            "avg_potential_loss": np.mean(potential_losses) if potential_losses else 0,
            "milestones_this_episode": sum(1 for m in self.milestones if m.achieved),
        }

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
            "milestone_bonus": self.milestone_bonus,
            "shaping_weight": self.shaping_weight,
            "policy": self.policy.to_dict(),
            "potential_critic": {
                "network": {
                    "layer_sizes": self.potential_critic.network.layer_sizes,
                    "weights": [w.tolist() for w in self.potential_critic.network.weights],
                    "biases": [b.tolist() for b in self.potential_critic.network.biases],
                },
            },
            "training_stats": self.training_stats,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MiRAAgent":
        """Load agent from file."""
        import json
        from pathlib import Path

        with open(Path(path)) as f:
            data = json.load(f)

        agent = cls(
            state_dim=data["state_dim"],
            hidden_sizes=data["hidden_sizes"],
            gamma=data["gamma"],
            milestone_bonus=data["milestone_bonus"],
            shaping_weight=data["shaping_weight"],
        )
        agent.policy = ActorCritic.from_dict(data["policy"])
        agent.training_stats = data.get("training_stats", agent.training_stats)

        # Load potential critic
        pc_data = data.get("potential_critic", {})
        if "network" in pc_data:
            agent.potential_critic.network = NeuralNetwork.from_dict(pc_data["network"])
            agent.potential_critic._sync_target()

        return agent
