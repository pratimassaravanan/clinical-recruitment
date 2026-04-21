"""Trainable offline policies for long-horizon clinical recruitment."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from inference import _best_available_patient, _best_site, _default_memory_key, _default_memory_payload, _default_memory_query, _estimated_token_cost, _infer_confidence, _infer_hypothesis, _normalize_action, _plan_summary_for_phase, _recommended_phase, PolicyState, rule_based_action


ACTIONS = [
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
class TrainingExample:
    features: Dict[str, float]
    action_type: str
    total_reward: float


def extract_features(obs: Dict[str, Any], step_num: int) -> Dict[str, float]:
    target = max(1, int(obs.get("target_enrollment", 1) or 1))
    enrolled = int(obs.get("enrolled_so_far", 0) or 0)
    budget = float(obs.get("budget_remaining", 0.0) or 0.0)
    token_budget_remaining = int(obs.get("token_budget_remaining", 0) or 0)
    token_usage = int(obs.get("token_usage_so_far", 0) or 0)
    memory = obs.get("patient_memory_summary", {})
    constraints = obs.get("active_constraints", {})
    return {
        "bias": 1.0,
        "progress": enrolled / target,
        "budget_scaled": min(1.0, budget / 150000.0),
        "time_scaled": min(1.0, int(obs.get("time_to_deadline_days", 0) or 0) / 180.0),
        "uncertainty": float(obs.get("uncertainty_level", 0.0) or 0.0),
        "dropout": float(obs.get("dropout_rate_7d", 0.0) or 0.0),
        "milestone_potential": float(obs.get("milestone_potential", 0.0) or 0.0),
        "token_efficiency": float(obs.get("token_efficiency_score", 1.0) or 0.0),
        "token_budget_scaled": min(1.0, token_budget_remaining / 12000.0),
        "token_usage_scaled": min(1.0, token_usage / 12000.0),
        "consented_pending": min(1.0, int(memory.get("consented_pending_allocation", 0)) / 5.0),
        "followup_due": min(1.0, int(memory.get("followup_due", 0)) / 5.0),
        "eligible_pending": min(1.0, int(memory.get("eligible_pending_consent", 0)) / 5.0),
        "reg_hold": min(1.0, int(constraints.get("regulatory_hold_days", 0)) / 5.0),
        "site_bottleneck": 1.0 if constraints.get("site_bottleneck", False) else 0.0,
        "step_scaled": min(1.0, step_num / 180.0),
    }


class LinearPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.weights: Dict[str, Dict[str, float]] = {
            action_type: {} for action_type in ACTIONS
        }

    def score(self, features: Dict[str, float], action_type: str) -> float:
        weights = self.weights.setdefault(action_type, {})
        return sum(weights.get(name, 0.0) * value for name, value in features.items())

    def choose_action_type(self, features: Dict[str, float], epsilon: float = 0.1) -> str:
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        scored = [(self.score(features, action_type), action_type) for action_type in ACTIONS]
        scored.sort(reverse=True)
        return scored[0][1]

    def update(self, examples: List[TrainingExample], learning_rate: float = 0.05) -> None:
        if not examples:
            return
        baseline = sum(example.total_reward for example in examples) / max(1, len(examples))
        for example in examples:
            advantage = example.total_reward - baseline
            weights = self.weights.setdefault(example.action_type, {})
            for name, value in example.features.items():
                weights[name] = weights.get(name, 0.0) + learning_rate * advantage * value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.weights, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearPolicy":
        policy = cls()
        policy.weights = json.loads(path.read_text(encoding="utf-8"))
        return policy


class MLPPolicy:
    """Small pure-Python two-layer policy model."""

    def __init__(self, seed: int = 0, hidden_size: int = 12):
        self.rng = random.Random(seed)
        self.hidden_size = hidden_size
        self.input_names = []
        self.hidden_weights: List[Dict[str, float]] = [dict() for _ in range(hidden_size)]
        self.hidden_biases: List[float] = [self.rng.uniform(-0.05, 0.05) for _ in range(hidden_size)]
        self.output_weights: Dict[str, List[float]] = {
            action_type: [self.rng.uniform(-0.05, 0.05) for _ in range(hidden_size)]
            for action_type in ACTIONS
        }
        self.output_biases: Dict[str, float] = {
            action_type: self.rng.uniform(-0.05, 0.05) for action_type in ACTIONS
        }

    def _ensure_inputs(self, features: Dict[str, float]) -> None:
        if self.input_names:
            return
        self.input_names = list(features.keys())
        for hidden_idx in range(self.hidden_size):
            self.hidden_weights[hidden_idx] = {
                name: self.rng.uniform(-0.05, 0.05) for name in self.input_names
            }

    def _forward(self, features: Dict[str, float]) -> tuple[List[float], Dict[str, float]]:
        self._ensure_inputs(features)
        hidden_values: List[float] = []
        for hidden_idx in range(self.hidden_size):
            total = self.hidden_biases[hidden_idx]
            for name, value in features.items():
                total += self.hidden_weights[hidden_idx].get(name, 0.0) * value
            hidden_values.append(max(0.0, total))

        outputs: Dict[str, float] = {}
        for action_type in ACTIONS:
            total = self.output_biases[action_type]
            for hidden_idx, hidden_value in enumerate(hidden_values):
                total += self.output_weights[action_type][hidden_idx] * hidden_value
            outputs[action_type] = total
        return hidden_values, outputs

    def score(self, features: Dict[str, float], action_type: str) -> float:
        _, outputs = self._forward(features)
        return outputs[action_type]

    def choose_action_type(self, features: Dict[str, float], epsilon: float = 0.1) -> str:
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        _, outputs = self._forward(features)
        return max(outputs.items(), key=lambda item: item[1])[0]

    def update(self, examples: List[TrainingExample], learning_rate: float = 0.02) -> None:
        if not examples:
            return
        baseline = sum(example.total_reward for example in examples) / max(1, len(examples))
        for example in examples:
            hidden_values, outputs = self._forward(example.features)
            chosen = example.action_type
            target = example.total_reward - baseline
            error = target - outputs.get(chosen, 0.0)
            for hidden_idx, hidden_value in enumerate(hidden_values):
                self.output_weights[chosen][hidden_idx] += learning_rate * error * hidden_value
            self.output_biases[chosen] += learning_rate * error
            for hidden_idx, hidden_value in enumerate(hidden_values):
                if hidden_value <= 0.0:
                    continue
                backprop = error * self.output_weights[chosen][hidden_idx]
                for name, value in example.features.items():
                    self.hidden_weights[hidden_idx][name] = (
                        self.hidden_weights[hidden_idx].get(name, 0.0)
                        + learning_rate * backprop * value
                    )
                self.hidden_biases[hidden_idx] += learning_rate * backprop

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_size": self.hidden_size,
            "input_names": self.input_names,
            "hidden_weights": self.hidden_weights,
            "hidden_biases": self.hidden_biases,
            "output_weights": self.output_weights,
            "output_biases": self.output_biases,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def action_payload_for_type(
    obs: Dict[str, Any],
    step_num: int,
    policy_state: PolicyState,
    action_type: str,
) -> Dict[str, Any]:
    hypothesis = _infer_hypothesis(obs)
    confidence = _infer_confidence(obs, step_num)
    if action_type == "screen_patient":
        patient_id = _best_available_patient(obs)
        payload = {
            "action_type": action_type,
            "patient_id": patient_id,
            "site_id": None,
            "strategy_change": None,
        }
    elif action_type == "recontact":
        candidates = policy_state.recontact_candidate_ids(step_num)
        payload = {
            "action_type": action_type,
            "patient_id": candidates[0] if candidates else None,
            "site_id": None,
            "strategy_change": None,
        }
    elif action_type == "allocate_to_site":
        candidates = policy_state.consented_pending_ids(step_num)
        payload = {
            "action_type": action_type,
            "patient_id": candidates[0] if candidates else None,
            "site_id": _best_site(obs, mode="allocate"),
            "strategy_change": None,
        }
    elif action_type == "adjust_strategy":
        recommended_phase = _recommended_phase(obs)
        if recommended_phase == "recovery" and _best_site(obs, mode="negotiate"):
            strategy_change = f"negotiate_site_{str(_best_site(obs, mode='negotiate')).replace('site_', '')}"
        elif float(obs.get("uncertainty_level", 0.0) or 0.0) > 0.45:
            strategy_change = "tighten_criteria"
        else:
            strategy_change = "increase_outreach"
        payload = {
            "action_type": action_type,
            "patient_id": None,
            "site_id": None,
            "strategy_change": strategy_change,
        }
    elif action_type == "plan_next_phase":
        phase = _recommended_phase(obs)
        payload = {
            "action_type": action_type,
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
            "plan_id": f"train-plan-{step_num}-{phase}",
            "plan_summary": _plan_summary_for_phase(phase, obs),
            "target_phase": phase,
        }
    elif action_type == "summarize_and_index":
        payload = {
            "action_type": action_type,
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
            "memory_key": _default_memory_key(obs),
            "memory_payload": _default_memory_payload(obs),
        }
    elif action_type == "retrieve_relevant_history":
        payload = {
            "action_type": action_type,
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
            "memory_query": _default_memory_query(obs),
        }
    else:
        payload = {
            "action_type": "stop_recruitment",
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
        }

    payload["hypothesis"] = hypothesis
    payload["confidence"] = confidence
    payload["token_cost"] = _estimated_token_cost(payload["action_type"])
    normalized = _normalize_action(payload, obs, step_num, policy_state)
    if normalized is not None:
        return normalized
    fallback = rule_based_action(obs, step_num, policy_state)
    normalized = _normalize_action(fallback, obs, step_num, policy_state)
    if normalized is None:
        raise RuntimeError(f"Failed to produce action for {action_type}")
    return normalized
