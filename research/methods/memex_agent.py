"""MemexRL: Memory-Augmented Reinforcement Learning.

Real implementation of episodic memory for long-horizon RL with:
1. Learned memory write operations - what to store and when
2. Learned memory read operations - what to retrieve and when
3. Memory-augmented value function
4. Retrieval-based action selection
"""

from __future__ import annotations

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
class MemoryEntry:
    """An entry in episodic memory."""

    key: np.ndarray  # Embedding for retrieval
    value: Dict[str, Any]  # Stored content
    step: int  # When it was stored
    importance: float  # Learned importance weight
    access_count: int = 0
    last_access: int = 0


class EpisodicMemory:
    """Differentiable episodic memory with learned read/write."""

    def __init__(
        self,
        key_dim: int = 32,
        value_dim: int = 64,
        max_entries: int = 100,
        state_dim: int = STATE_DIM,
    ):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_entries = max_entries
        self.state_dim = state_dim

        # Key encoder: state -> key embedding
        self.key_encoder = NeuralNetwork(
            layer_sizes=[state_dim, 64, key_dim],
            learning_rate=0.001,
        )

        # Value encoder: state + action + reward -> value embedding
        self.value_encoder = NeuralNetwork(
            layer_sizes=[state_dim + len(ACTION_SPACE) + 1, 64, value_dim],
            learning_rate=0.001,
        )

        # Write gate: decides whether to write
        self.write_gate = NeuralNetwork(
            layer_sizes=[state_dim, 32, 1],
            learning_rate=0.001,
        )

        # Importance predictor: how important is this memory
        self.importance_net = NeuralNetwork(
            layer_sizes=[state_dim, 32, 1],
            learning_rate=0.001,
        )

        # Memory storage
        self.entries: List[MemoryEntry] = []

        # Stats
        self.write_count = 0
        self.read_count = 0

    def reset(self):
        """Clear memory for new episode."""
        self.entries = []

    def encode_key(self, state: np.ndarray) -> np.ndarray:
        """Encode state into key embedding."""
        key, _ = self.key_encoder.forward(state)
        # Normalize to unit sphere for cosine similarity
        key = key / (np.linalg.norm(key) + 1e-8)
        return key

    def encode_value(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
    ) -> np.ndarray:
        """Encode state-action-reward into value embedding."""
        action_onehot = np.zeros(len(ACTION_SPACE), dtype=np.float32)
        action_onehot[action] = 1.0
        input_vec = np.concatenate([state, action_onehot, [reward]])
        value, _ = self.value_encoder.forward(input_vec)
        return value

    def compute_write_probability(self, state: np.ndarray) -> float:
        """Compute probability of writing to memory."""
        logit, _ = self.write_gate.forward(state)
        prob = 1.0 / (1.0 + np.exp(-float(logit[0] if logit.ndim > 0 else logit)))
        return prob

    def compute_importance(self, state: np.ndarray) -> float:
        """Compute importance score for memory entry."""
        score, _ = self.importance_net.forward(state)
        # Sigmoid to [0, 1]
        importance = 1.0 / (1.0 + np.exp(-float(score[0] if score.ndim > 0 else score)))
        return importance

    def write(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        step: int,
        obs: Dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Attempt to write to memory. Returns True if written."""
        write_prob = self.compute_write_probability(state)

        # Stochastic write decision (or forced)
        if not force and np.random.random() > write_prob:
            return False

        key = self.encode_key(state)
        value_emb = self.encode_value(state, action, reward)
        importance = self.compute_importance(state)

        entry = MemoryEntry(
            key=key,
            value={
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "obs_summary": {
                    "enrolled": obs.get("enrolled_so_far", 0),
                    "budget": obs.get("budget_remaining", 0),
                    "milestone_potential": obs.get("milestone_potential", 0),
                },
                "value_embedding": value_emb.tolist(),
            },
            step=step,
            importance=importance,
        )

        self.entries.append(entry)
        self.write_count += 1

        # Evict low-importance entries if over capacity
        if len(self.entries) > self.max_entries:
            self._evict_lowest_importance()

        return True

    def _evict_lowest_importance(self):
        """Remove lowest importance entry."""
        if not self.entries:
            return

        # Score combines importance and recency
        scores = []
        for i, entry in enumerate(self.entries):
            recency = 1.0 / (1.0 + self.write_count - entry.step)
            access_bonus = 0.1 * entry.access_count
            score = entry.importance * 0.5 + recency * 0.3 + access_bonus * 0.2
            scores.append((score, i))

        scores.sort()
        # Remove lowest scoring entry
        _, idx = scores[0]
        self.entries.pop(idx)

    def read(
        self,
        query_state: np.ndarray,
        top_k: int = 5,
        current_step: int = 0,
    ) -> Tuple[np.ndarray, List[MemoryEntry]]:
        """Read from memory using attention-based retrieval."""
        if not self.entries:
            return np.zeros(self.value_dim, dtype=np.float32), []

        query_key = self.encode_key(query_state)

        # Compute attention scores (cosine similarity)
        scores = []
        for entry in self.entries:
            similarity = float(np.dot(query_key, entry.key))
            # Boost by importance
            score = similarity * (0.5 + 0.5 * entry.importance)
            scores.append(score)

        scores = np.array(scores)

        # Softmax attention
        attention = _softmax(scores * 5.0)  # Temperature scaling

        # Get top-k entries
        top_indices = np.argsort(scores)[-top_k:]
        retrieved_entries = [self.entries[i] for i in top_indices]

        # Update access stats
        for i in top_indices:
            self.entries[i].access_count += 1
            self.entries[i].last_access = current_step

        # Weighted sum of value embeddings
        retrieved_value = np.zeros(self.value_dim, dtype=np.float32)
        for i, entry in enumerate(self.entries):
            value_emb = np.array(entry.value.get("value_embedding", np.zeros(self.value_dim)))
            retrieved_value += attention[i] * value_emb

        self.read_count += 1

        return retrieved_value, retrieved_entries

    def update_write_gate(
        self,
        state: np.ndarray,
        should_write: bool,
    ):
        """Update write gate based on hindsight."""
        logit, activations = self.write_gate.forward(state)
        prob = 1.0 / (1.0 + np.exp(-float(logit[0] if logit.ndim > 0 else logit)))

        # Binary cross-entropy gradient
        target = 1.0 if should_write else 0.0
        grad = np.array([prob - target])

        weight_grads, bias_grads = self.write_gate.backward(activations, grad)
        self.write_gate.update(weight_grads, bias_grads)

    def update_importance(
        self,
        state: np.ndarray,
        actual_importance: float,
    ):
        """Update importance predictor."""
        pred, activations = self.importance_net.forward(state)
        pred_val = 1.0 / (1.0 + np.exp(-float(pred[0] if pred.ndim > 0 else pred)))

        # MSE gradient
        grad = np.array([pred_val - actual_importance])

        weight_grads, bias_grads = self.importance_net.backward(activations, grad)
        self.importance_net.update(weight_grads, bias_grads)


class MemexRLAgent:
    """MemexRL: Memory-augmented RL agent.

    Implements:
    - Learned memory write policy
    - Attention-based memory retrieval
    - Memory-augmented value function
    - Retrieval-guided action selection
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_sizes: Optional[List[int]] = None,
        gamma: float = 0.99,
        memory_key_dim: int = 32,
        memory_value_dim: int = 64,
        memory_size: int = 100,
    ):
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.gamma = gamma
        self.memory_key_dim = memory_key_dim
        self.memory_value_dim = memory_value_dim
        self.memory_size = memory_size

        # Episodic memory
        self.memory = EpisodicMemory(
            key_dim=memory_key_dim,
            value_dim=memory_value_dim,
            max_entries=memory_size,
            state_dim=state_dim,
        )

        # Memory-augmented policy (state + retrieved memory)
        augmented_dim = state_dim + memory_value_dim
        self.policy = ActorCritic(
            state_dim=augmented_dim,
            action_dim=len(ACTION_SPACE),
            hidden_sizes=self.hidden_sizes,
            gamma=gamma,
        )

        # Memory-augmented value function
        self.memory_value_net = NeuralNetwork(
            layer_sizes=[augmented_dim, 64, 32, 1],
            learning_rate=0.001,
        )

        # Training stats
        self.training_stats = {
            "episodes": 0,
            "memory_writes": 0,
            "memory_reads": 0,
            "avg_retrieval_similarity": 0.0,
        }

    def reset(self):
        """Reset for new episode."""
        self.memory.reset()

    def select_action(
        self,
        obs: Dict[str, Any],
        step: int,
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using memory-augmented policy."""
        state = extract_state_features(obs)

        # Read from memory
        retrieved_value, retrieved_entries = self.memory.read(
            state, top_k=5, current_step=step
        )
        self.training_stats["memory_reads"] += 1

        # Augment state with retrieved memory
        augmented_state = np.concatenate([state, retrieved_value])

        action = self.policy.select_action(augmented_state, deterministic)

        info = {
            "state_value": self.policy.get_value(augmented_state),
            "memory_size": len(self.memory.entries),
            "retrieved_count": len(retrieved_entries),
            "memory_value_norm": float(np.linalg.norm(retrieved_value)),
        }

        return action, info

    def step(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        step: int,
    ):
        """Process a step and potentially write to memory."""
        state = extract_state_features(obs)

        # Decide whether to write based on learned gate
        written = self.memory.write(
            state, action, reward, step, obs
        )

        if written:
            self.training_stats["memory_writes"] += 1

    def _compute_hindsight_importance(
        self,
        entry_step: int,
        trajectory: List[Dict[str, Any]],
    ) -> float:
        """Compute hindsight importance of a memory entry."""
        if entry_step >= len(trajectory):
            return 0.0

        # Importance = future discounted return from this step
        future_return = 0.0
        discount = 1.0
        for i in range(entry_step, len(trajectory)):
            future_return += discount * float(trajectory[i].get("reward", 0))
            discount *= self.gamma

        # Normalize to [0, 1]
        normalized = 1.0 / (1.0 + np.exp(-future_return))
        return normalized

    def update_from_episode(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Update policy and memory from episode."""
        if len(trajectory) < 2:
            return {}

        # Rebuild memory and collect training data
        self.reset()
        augmented_states = []
        actions = []
        rewards = []
        dones = []

        for i, t in enumerate(trajectory):
            obs = t.get("obs", {})
            state = extract_state_features(obs)
            action = t.get("action", 0)
            reward = float(t.get("reward", 0))
            done = t.get("done", False)

            # Read from memory before action
            retrieved_value, _ = self.memory.read(state, top_k=5, current_step=i)
            augmented_state = np.concatenate([state, retrieved_value])

            augmented_states.append(augmented_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            # Write to memory after action
            self.memory.write(state, action, reward, i, obs)

        # Policy gradient update
        policy_loss = self.policy.update_from_trajectory(
            augmented_states, actions, rewards, dones
        )

        # Update write gate and importance predictor with hindsight
        for entry in self.memory.entries:
            state = np.array(entry.value.get("state", np.zeros(self.state_dim)))

            # Hindsight importance
            actual_importance = self._compute_hindsight_importance(
                entry.step, trajectory
            )
            self.memory.update_importance(state, actual_importance)

            # Hindsight write decision: was this a good write?
            # Good if importance > median
            median_importance = np.median([e.importance for e in self.memory.entries])
            should_have_written = actual_importance > median_importance
            self.memory.update_write_gate(state, should_have_written)

        self.training_stats["episodes"] += 1

        return {
            "policy_actor_loss": policy_loss.get("actor_loss", 0),
            "policy_critic_loss": policy_loss.get("critic_loss", 0),
            "memory_writes": self.training_stats["memory_writes"],
            "memory_reads": self.training_stats["memory_reads"],
            "final_memory_size": len(self.memory.entries),
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
            "memory_key_dim": self.memory_key_dim,
            "memory_value_dim": self.memory_value_dim,
            "memory_size": self.memory_size,
            "policy": self.policy.to_dict(),
            "memory": {
                "key_encoder": {
                    "layer_sizes": self.memory.key_encoder.layer_sizes,
                    "weights": [w.tolist() for w in self.memory.key_encoder.weights],
                    "biases": [b.tolist() for b in self.memory.key_encoder.biases],
                },
                "value_encoder": {
                    "layer_sizes": self.memory.value_encoder.layer_sizes,
                    "weights": [w.tolist() for w in self.memory.value_encoder.weights],
                    "biases": [b.tolist() for b in self.memory.value_encoder.biases],
                },
                "write_gate": {
                    "layer_sizes": self.memory.write_gate.layer_sizes,
                    "weights": [w.tolist() for w in self.memory.write_gate.weights],
                    "biases": [b.tolist() for b in self.memory.write_gate.biases],
                },
                "importance_net": {
                    "layer_sizes": self.memory.importance_net.layer_sizes,
                    "weights": [w.tolist() for w in self.memory.importance_net.weights],
                    "biases": [b.tolist() for b in self.memory.importance_net.biases],
                },
            },
            "training_stats": self.training_stats,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MemexRLAgent":
        """Load agent from file."""
        import json
        from pathlib import Path

        with open(Path(path)) as f:
            data = json.load(f)

        agent = cls(
            state_dim=data["state_dim"],
            hidden_sizes=data["hidden_sizes"],
            gamma=data["gamma"],
            memory_key_dim=data["memory_key_dim"],
            memory_value_dim=data["memory_value_dim"],
            memory_size=data["memory_size"],
        )
        agent.policy = ActorCritic.from_dict(data["policy"])
        agent.training_stats = data.get("training_stats", agent.training_stats)

        # Load memory networks
        mem_data = data.get("memory", {})
        if "key_encoder" in mem_data:
            agent.memory.key_encoder = NeuralNetwork.from_dict(mem_data["key_encoder"])
        if "value_encoder" in mem_data:
            agent.memory.value_encoder = NeuralNetwork.from_dict(mem_data["value_encoder"])
        if "write_gate" in mem_data:
            agent.memory.write_gate = NeuralNetwork.from_dict(mem_data["write_gate"])
        if "importance_net" in mem_data:
            agent.memory.importance_net = NeuralNetwork.from_dict(mem_data["importance_net"])

        return agent
