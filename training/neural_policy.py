"""Neural network policy with real backpropagation training.

Implements a proper actor-critic architecture with:
- Policy network (actor) for action selection
- Value network (critic) for state value estimation
- Advantage estimation for policy gradient updates
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / (exp_x.sum() + 1e-8)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


ACTION_SPACE = [
    "screen_patient",
    "recontact",
    "allocate_to_site",
    "adjust_strategy",
    "plan_next_phase",
    "summarize_and_index",
    "retrieve_relevant_history",
    "stop_recruitment",
]


@dataclass
class NeuralNetwork:
    """Simple feedforward neural network with backpropagation."""

    layer_sizes: List[int]
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[np.ndarray] = field(default_factory=list)
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        if not self.weights:
            self._init_weights()

    def _init_weights(self):
        """Xavier initialization for weights."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = math.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass returning output and intermediate activations."""
        activations = [x]
        current = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            if i < len(self.weights) - 1:
                current = _relu(z)
            else:
                current = z  # Linear output for critic, softmax applied separately for actor
            activations.append(current)
        return current, activations

    def backward(
        self,
        activations: List[np.ndarray],
        output_grad: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass computing gradients."""
        weight_grads = []
        bias_grads = []
        delta = output_grad

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            a_prev = activations[i]
            if a_prev.ndim == 1:
                a_prev = a_prev.reshape(1, -1)
            if delta.ndim == 1:
                delta = delta.reshape(1, -1)

            dw = a_prev.T @ delta + self.weight_decay * self.weights[i]
            db = delta.sum(axis=0)
            weight_grads.insert(0, dw)
            bias_grads.insert(0, db)

            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU gradient
                z_prev = activations[i]
                if z_prev.ndim == 1:
                    z_prev = z_prev.reshape(1, -1)
                delta = delta * _relu_grad(z_prev)

        return weight_grads, bias_grads

    def update(self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray]):
        """Apply gradient descent update with clipping."""
        for i in range(len(self.weights)):
            # Gradient clipping
            w_grad = np.clip(weight_grads[i], -1.0, 1.0)
            b_grad = np.clip(bias_grads[i], -1.0, 1.0)
            self.weights[i] -= self.learning_rate * w_grad
            self.biases[i] -= self.learning_rate * b_grad
            # Clip weights to prevent explosion
            self.weights[i] = np.clip(self.weights[i], -10.0, 10.0)
            self.biases[i] = np.clip(self.biases[i], -10.0, 10.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuralNetwork":
        nn = cls(layer_sizes=data["layer_sizes"], learning_rate=data.get("learning_rate", 0.001))
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        return nn


@dataclass
class ActorCritic:
    """Actor-Critic policy with separate policy and value networks."""

    state_dim: int
    action_dim: int = len(ACTION_SPACE)
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    actor: Optional[NeuralNetwork] = None
    critic: Optional[NeuralNetwork] = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    def __post_init__(self):
        if self.actor is None:
            actor_sizes = [self.state_dim] + self.hidden_sizes + [self.action_dim]
            self.actor = NeuralNetwork(layer_sizes=actor_sizes)
        if self.critic is None:
            critic_sizes = [self.state_dim] + self.hidden_sizes + [1]
            self.critic = NeuralNetwork(layer_sizes=critic_sizes)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities from policy network."""
        logits, _ = self.actor.forward(state)
        return _softmax(logits)

    def get_value(self, state: np.ndarray) -> float:
        """Get state value from critic network."""
        value, _ = self.critic.forward(state)
        return float(value[0] if value.ndim > 0 else value)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action using policy."""
        probs = self.get_action_probs(state)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0.0

        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update_from_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Update policy and value networks from a trajectory."""
        if len(states) < 2:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # Compute values for all states
        values = [self.get_value(s) for s in states]
        next_value = 0.0 if dones[-1] else self.get_value(states[-1])

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values[:-1], dones, next_value)

        # Normalize advantages
        adv_array = np.array(advantages)
        if len(adv_array) > 1:
            adv_array = (adv_array - adv_array.mean()) / (adv_array.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        # Update networks for each timestep
        for t in range(len(states) - 1):
            state = states[t]
            action = actions[t]
            advantage = adv_array[t]
            target_return = returns[t]

            # Actor update
            logits, actor_activations = self.actor.forward(state)
            probs = _softmax(logits)
            log_prob = math.log(probs[action] + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            # Policy gradient: -log_prob * advantage
            actor_loss = -log_prob * advantage - self.entropy_coef * entropy
            total_actor_loss += actor_loss
            total_entropy += entropy

            # Compute actor gradient
            grad_logits = probs.copy()
            grad_logits[action] -= 1.0
            grad_logits *= -advantage
            # Add entropy gradient
            grad_logits += self.entropy_coef * (np.log(probs + 1e-8) + 1.0)

            actor_weight_grads, actor_bias_grads = self.actor.backward(actor_activations, grad_logits)
            self.actor.update(actor_weight_grads, actor_bias_grads)

            # Critic update
            value, critic_activations = self.critic.forward(state)
            value = float(value[0] if value.ndim > 0 else value)
            critic_loss = 0.5 * (value - target_return) ** 2
            total_critic_loss += critic_loss

            # Critic gradient
            grad_value = np.array([value - target_return])
            critic_weight_grads, critic_bias_grads = self.critic.backward(critic_activations, grad_value)
            self.critic.update(critic_weight_grads, critic_bias_grads)

        n = len(states) - 1
        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
            "actor": self.actor.to_dict(),
            "critic": self.critic.to_dict(),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActorCritic":
        ac = cls(
            state_dim=data["state_dim"],
            action_dim=data.get("action_dim", len(ACTION_SPACE)),
            hidden_sizes=data.get("hidden_sizes", [64, 32]),
            gamma=data.get("gamma", 0.99),
            gae_lambda=data.get("gae_lambda", 0.95),
            entropy_coef=data.get("entropy_coef", 0.01),
            value_coef=data.get("value_coef", 0.5),
        )
        ac.actor = NeuralNetwork.from_dict(data["actor"])
        ac.critic = NeuralNetwork.from_dict(data["critic"])
        return ac

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ActorCritic":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def extract_state_features(obs: Dict[str, Any]) -> np.ndarray:
    """Extract numerical feature vector from observation dict."""
    max_steps = float(
        obs.get(
            "max_steps",
            max(1.0, float(obs.get("timestamp", 0) or 0) + float(obs.get("time_to_deadline_days", 180) or 0)),
        )
        or 1.0
    )
    initial_budget = max(1.0, float(obs.get("initial_budget", 150000.0) or 150000.0))
    time_to_deadline_days = float(
        obs.get("time_to_deadline_days", max(0.0, max_steps - float(obs.get("timestamp", 0) or 0)))
        or 0.0
    )
    features = [
        float(obs.get("timestamp", 0) or 0) / max_steps,
        float(obs.get("budget_remaining", 0) or 0) / initial_budget,
        time_to_deadline_days / max_steps,
        obs.get("enrolled_so_far", 0) / 150.0,
        obs.get("target_enrollment", 100) / 150.0,
        obs.get("uncertainty_level", 0.0),
        obs.get("dropout_rate_7d", 0.0),
        obs.get("screening_backlog", 0) / 10.0,
        obs.get("milestone_potential", 0.0),
        obs.get("token_efficiency_score", 1.0),
        obs.get("hypothesis_accuracy", 0.0),
        len(obs.get("available_patients", [])) / 5.0,
    ]

    # Funnel features
    funnel = obs.get("current_funnel", {})
    features.extend([
        funnel.get("screened", 0) / 200.0,
        funnel.get("eligible", 0) / 150.0,
        funnel.get("consented", 0) / 100.0,
        funnel.get("enrolled", 0) / 100.0,
        funnel.get("dropped", 0) / 50.0,
    ])

    # Milestone features
    milestones = obs.get("milestones", {})
    features.extend([
        float(milestones.get("25pct", False)),
        float(milestones.get("50pct", False)),
        float(milestones.get("75pct", False)),
        float(milestones.get("100pct", False)),
    ])

    # Constraint features
    constraints = obs.get("active_constraints", {})
    features.extend([
        float(constraints.get("regulatory_hold_days", 0)) / 10.0,
        float(constraints.get("competitor_pressure", 0.0)),
        float(constraints.get("sentiment_pressure", 0.0)),
        float(constraints.get("sponsor_pressure", False)),
        float(constraints.get("backlog_pressure", False)),
        float(constraints.get("site_bottleneck", False)),
    ])

    # Uncertainty components
    unc = obs.get("uncertainty_components", {})
    features.extend([
        float(unc.get("patient_pool", 0.0)),
        float(unc.get("site_operations", 0.0)),
        float(unc.get("policy", 0.0)),
    ])

    # Patient memory
    mem = obs.get("patient_memory_summary", {})
    features.extend([
        mem.get("high_priority_candidates", 0) / 20.0,
        mem.get("eligible_pending_consent", 0) / 20.0,
        mem.get("consented_pending_allocation", 0) / 20.0,
        mem.get("at_risk_enrolled", 0) / 20.0,
        mem.get("followup_due", 0) / 10.0,
    ])

    # Counterfactual rollout
    cf = obs.get("counterfactual_rollout", {})
    features.extend([
        float(cf.get("allocate_gain_estimate", 0.0)),
        float(cf.get("recontact_gain_estimate", 0.0)),
    ])

    return np.array(features, dtype=np.float32)


STATE_DIM = 37  # Must match extract_state_features output length
