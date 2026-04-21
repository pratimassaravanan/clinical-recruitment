"""KLong: Long-context Trajectory Processing with Temporal Credit Assignment.

Real implementation of long-horizon credit assignment with:
1. Trajectory segmentation with overlap for context preservation
2. Temporal difference learning with eligibility traces
3. Multi-scale temporal abstraction
4. Context-aware value function
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
    extract_state_features,
)


@dataclass
class TrajectorySegment:
    """A segment of trajectory with context."""

    segment_id: int
    start_step: int
    end_step: int
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    context_embedding: Optional[np.ndarray] = None
    segment_return: float = 0.0


class TemporalAbstraction:
    """Multi-scale temporal abstraction for long horizons."""

    def __init__(
        self,
        scales: List[int] = None,  # Time scales: [1, 5, 20, 60]
        state_dim: int = STATE_DIM,
    ):
        self.scales = scales or [1, 5, 20, 60]
        self.state_dim = state_dim

        # Aggregation networks for each scale
        self.aggregators = {}
        for scale in self.scales:
            self.aggregators[scale] = NeuralNetwork(
                layer_sizes=[state_dim * min(scale, 10), 32, state_dim],
                learning_rate=0.001,
            )

    def aggregate_at_scale(
        self,
        states: List[np.ndarray],
        scale: int,
    ) -> np.ndarray:
        """Aggregate states at a given temporal scale."""
        if len(states) < scale:
            # Pad with last state
            states = states + [states[-1]] * (scale - len(states))

        # Take last `scale` states (or all if fewer)
        recent = states[-min(scale, 10):]
        # Concatenate and pass through aggregator
        concat = np.concatenate(recent)
        output, _ = self.aggregators[scale].forward(concat)
        return output

    def get_multiscale_embedding(
        self,
        states: List[np.ndarray],
    ) -> np.ndarray:
        """Get multi-scale temporal embedding."""
        embeddings = []
        for scale in self.scales:
            emb = self.aggregate_at_scale(states, scale)
            embeddings.append(emb)

        # Concatenate all scale embeddings
        return np.concatenate(embeddings)


class EligibilityTraces:
    """Eligibility traces for temporal credit assignment."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lambda_: float = 0.9,
        gamma: float = 0.99,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lambda_ = lambda_
        self.gamma = gamma

        # Traces for state-action pairs (accumulated gradients)
        self.traces: Dict[str, np.ndarray] = {}

    def reset(self):
        """Reset traces for new episode."""
        self.traces = {}

    def update_trace(self, state_key: str, gradient: np.ndarray):
        """Update eligibility trace with new gradient."""
        if state_key not in self.traces:
            self.traces[state_key] = np.zeros_like(gradient)

        # Accumulating trace: e(s,a) = gamma * lambda * e(s,a) + gradient
        self.traces[state_key] = (
            self.gamma * self.lambda_ * self.traces[state_key] + gradient
        )

    def get_trace(self, state_key: str) -> np.ndarray:
        """Get current trace value."""
        return self.traces.get(state_key, None)

    def decay_all(self):
        """Decay all traces."""
        for key in self.traces:
            self.traces[key] *= self.gamma * self.lambda_


class KLongAgent:
    """KLong: Long-context agent with temporal credit assignment.

    Implements:
    - Trajectory segmentation with overlapping windows
    - TD(lambda) with eligibility traces
    - Multi-scale temporal abstraction
    - Context-aware policy and value functions
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_sizes: Optional[List[int]] = None,
        gamma: float = 0.99,
        lambda_: float = 0.9,  # TD(lambda) parameter
        segment_length: int = 30,
        segment_overlap: int = 10,
        temporal_scales: Optional[List[int]] = None,
    ):
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.gamma = gamma
        self.lambda_ = lambda_
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.temporal_scales = temporal_scales or [1, 5, 20, 60]

        # Multi-scale temporal abstraction
        self.temporal_abstraction = TemporalAbstraction(
            scales=self.temporal_scales,
            state_dim=state_dim,
        )

        # Context embedding dimension
        self.context_dim = state_dim * len(self.temporal_scales)

        # Context-aware policy (state + context)
        self.policy = ActorCritic(
            state_dim=state_dim + self.context_dim,
            action_dim=len(ACTION_SPACE),
            hidden_sizes=self.hidden_sizes,
            gamma=gamma,
            gae_lambda=lambda_,
        )

        # Segment-level value function
        self.segment_critic = NeuralNetwork(
            layer_sizes=[self.context_dim, 64, 32, 1],
            learning_rate=0.001,
        )

        # Eligibility traces
        self.traces = EligibilityTraces(
            state_dim=state_dim + self.context_dim,
            action_dim=len(ACTION_SPACE),
            lambda_=lambda_,
            gamma=gamma,
        )

        # State history for context
        self.state_history: List[np.ndarray] = []

        # Training stats
        self.training_stats = {
            "episodes": 0,
            "segments_processed": 0,
            "avg_segment_return": 0.0,
            "trace_updates": 0,
        }

    def reset(self):
        """Reset for new episode."""
        self.state_history = []
        self.traces.reset()

    def _get_context_embedding(self) -> np.ndarray:
        """Get current context embedding from state history."""
        if not self.state_history:
            return np.zeros(self.context_dim, dtype=np.float32)
        return self.temporal_abstraction.get_multiscale_embedding(self.state_history)

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using context-aware policy."""
        state = extract_state_features(obs)
        self.state_history.append(state)

        # Get context embedding
        context = self._get_context_embedding()

        # Combine state with context
        augmented_state = np.concatenate([state, context])

        action = self.policy.select_action(augmented_state, deterministic)

        info = {
            "state_value": self.policy.get_value(augmented_state),
            "context_norm": float(np.linalg.norm(context)),
            "history_length": len(self.state_history),
        }

        return action, info

    def _segment_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> List[TrajectorySegment]:
        """Segment trajectory with overlap."""
        segments = []
        step = self.segment_length - self.segment_overlap
        segment_id = 0

        i = 0
        while i < len(trajectory):
            end_idx = min(i + self.segment_length, len(trajectory))

            segment = TrajectorySegment(
                segment_id=segment_id,
                start_step=i,
                end_step=end_idx,
            )

            # Extract segment data
            segment_states = []
            for j in range(i, end_idx):
                t = trajectory[j]
                obs = t.get("obs", {})
                state = extract_state_features(obs)
                segment.states.append(state)
                segment.actions.append(t.get("action", 0))
                segment.rewards.append(float(t.get("reward", 0)))
                segment_states.append(state)

            # Compute context embedding for segment
            if segment_states:
                segment.context_embedding = self.temporal_abstraction.get_multiscale_embedding(
                    segment_states
                )

            # Compute segment return
            segment.segment_return = sum(
                r * (self.gamma ** t)
                for t, r in enumerate(segment.rewards)
            )

            segments.append(segment)
            segment_id += 1
            i += step

            if end_idx >= len(trajectory):
                break

        return segments

    def _update_with_eligibility_traces(
        self,
        state: np.ndarray,
        action: int,
        td_error: float,
    ):
        """Update policy using eligibility traces."""
        # Get policy gradient for this state-action
        augmented_state = state
        logits, activations = self.policy.actor.forward(augmented_state)
        probs = np.exp(logits - np.max(logits))
        probs = probs / (probs.sum() + 1e-8)

        # Policy gradient direction
        grad = probs.copy()
        grad[action] -= 1.0

        # Update trace
        state_key = str(hash(state.tobytes()))
        self.traces.update_trace(state_key, grad)

        # Apply TD error scaled by trace
        trace = self.traces.get_trace(state_key)
        if trace is not None:
            # Scale gradient by TD error and trace
            update_grad = td_error * trace
            # Reshape for backward pass
            weight_grads, bias_grads = self.policy.actor.backward(activations, update_grad)
            self.policy.actor.update(weight_grads, bias_grads)
            self.training_stats["trace_updates"] += 1

        # Decay all traces
        self.traces.decay_all()

    def update_from_episode(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Update policy from episode using segmented learning."""
        if len(trajectory) < 2:
            return {}

        # Segment trajectory
        segments = self._segment_trajectory(trajectory)

        total_segment_return = 0.0
        segment_losses = []

        for segment in segments:
            if len(segment.states) < 2:
                continue

            self.training_stats["segments_processed"] += 1
            total_segment_return += segment.segment_return

            # Build augmented states with context
            augmented_states = []
            accumulated_states = []

            for i, state in enumerate(segment.states):
                accumulated_states.append(state)
                context = self.temporal_abstraction.get_multiscale_embedding(accumulated_states)
                augmented_state = np.concatenate([state, context])
                augmented_states.append(augmented_state)

            # Compute TD errors and update with eligibility traces
            for i in range(len(segment.states) - 1):
                current_value = self.policy.get_value(augmented_states[i])
                next_value = self.policy.get_value(augmented_states[i + 1])
                reward = segment.rewards[i]

                td_error = reward + self.gamma * next_value - current_value

                # Update with eligibility traces
                self._update_with_eligibility_traces(
                    augmented_states[i],
                    segment.actions[i],
                    td_error,
                )

            # Standard policy gradient update for segment
            if len(segment.rewards) >= 2:
                dones = [False] * len(segment.rewards)
                dones[-1] = True
                loss = self.policy.update_from_trajectory(
                    augmented_states[:len(segment.rewards)],
                    segment.actions[:len(segment.rewards) - 1],
                    segment.rewards[:-1],
                    dones[:-1],
                )
                segment_losses.append(loss)

            # Update segment critic
            if segment.context_embedding is not None:
                seg_value, seg_activations = self.segment_critic.forward(segment.context_embedding)
                seg_value = float(seg_value[0] if seg_value.ndim > 0 else seg_value)
                seg_target = segment.segment_return
                seg_error = seg_target - seg_value
                seg_grad = np.array([-seg_error])
                w_grads, b_grads = self.segment_critic.backward(seg_activations, seg_grad)
                self.segment_critic.update(w_grads, b_grads)

        self.training_stats["episodes"] += 1
        self.training_stats["avg_segment_return"] = (
            self.training_stats["avg_segment_return"] * 0.9
            + (total_segment_return / max(1, len(segments))) * 0.1
        )

        return {
            "segments_processed": len(segments),
            "avg_segment_return": total_segment_return / max(1, len(segments)),
            "avg_actor_loss": np.mean([l.get("actor_loss", 0) for l in segment_losses]) if segment_losses else 0,
            "trace_updates": self.training_stats["trace_updates"],
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
            "lambda_": self.lambda_,
            "segment_length": self.segment_length,
            "segment_overlap": self.segment_overlap,
            "temporal_scales": self.temporal_scales,
            "policy": self.policy.to_dict(),
            "segment_critic": {
                "layer_sizes": self.segment_critic.layer_sizes,
                "weights": [w.tolist() for w in self.segment_critic.weights],
                "biases": [b.tolist() for b in self.segment_critic.biases],
            },
            "training_stats": self.training_stats,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KLongAgent":
        """Load agent from file."""
        import json
        from pathlib import Path

        with open(Path(path)) as f:
            data = json.load(f)

        agent = cls(
            state_dim=data["state_dim"],
            hidden_sizes=data["hidden_sizes"],
            gamma=data["gamma"],
            lambda_=data["lambda_"],
            segment_length=data["segment_length"],
            segment_overlap=data["segment_overlap"],
            temporal_scales=data["temporal_scales"],
        )
        agent.policy = ActorCritic.from_dict(data["policy"])
        agent.training_stats = data.get("training_stats", agent.training_stats)

        # Load segment critic
        sc_data = data.get("segment_critic", {})
        if "layer_sizes" in sc_data:
            agent.segment_critic = NeuralNetwork.from_dict(sc_data)

        return agent
