"""TRL-compatible tool environment for GRPO training.

This exposes individual tool methods (screen_patient, recontact, allocate_to_site, etc.)
as required by TRL's `environment_factory` protocol. Each method has a typed signature
and Args docstring so the trainer can build tool schemas for the model.

Usage with TRL GRPOTrainer:
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        environment_factory=ClinicalRecruitmentToolEnv,
        ...
    )

See: https://huggingface.co/docs/trl/openenv
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from env import ClinicalRecruitmentEnv
from models import Action


class ClinicalRecruitmentToolEnv:
    """TRL environment_factory-compatible wrapper.

    Exposes each action type as a separate public method with docstrings and
    typed arguments, exactly as TRL's GRPOTrainer expects.

    Stores cumulative reward, enrollment state, and action history for use
    in reward functions via the `environments` parameter.
    """

    def __init__(self):
        self._env = ClinicalRecruitmentEnv()
        self.reward: float = 0.0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.last_observation: Optional[Dict[str, Any]] = None
        self.initial_budget: float = 0.0
        self.action_history: list[str] = []
        self.enrollment_history: list[int] = []
        self.budget_history: list[float] = []
        self.hypothesis_history: list[str] = []

    def reset(self, **kwargs) -> str | None:
        """Reset the environment for a new episode.

        Called at the start of each episode by GRPOTrainer.
        Receives dataset columns as kwargs (e.g., task_id).
        Returns initial observation text or None.
        """
        task = kwargs.get("task") or kwargs.get("task_id") or "easy_bench"
        result = self._env.reset(task=task)
        self.last_observation = result.observation.model_dump()
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.done = bool(result.done)
        self.action_history = []
        self.enrollment_history = []
        self.budget_history = []
        self.hypothesis_history = []

        obs = self.last_observation
        self.initial_budget = obs.get("budget_remaining", 0.0)
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0.0))

        return self._format_observation()

    # ------------------------------------------------------------------
    # Tool methods — each is a public method with Args docstring
    # ------------------------------------------------------------------

    def screen_patient(self, patient_id: str, hypothesis: str = "noise_dominant", confidence: float = 0.7) -> str:
        """Screen a candidate patient for trial eligibility.

        Args:
            patient_id: Patient ID from the available_patients list in the observation.
            hypothesis: Your hypothesis about which dynamic dominates: noise_dominant, dropout_dominant, site_bias, confounding, or unknown.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing screening result and new state.
        """
        return self._step(Action(
            action_type="screen_patient",
            patient_id=patient_id,
            hypothesis=hypothesis,
            confidence=confidence,
        ))

    def recontact(self, patient_id: str, hypothesis: str = "noise_dominant", confidence: float = 0.7) -> str:
        """Recontact a previously screened or eligible patient to obtain consent.

        Args:
            patient_id: Patient ID from the recontact_candidates list in the observation.
            hypothesis: Your hypothesis about which dynamic dominates.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing recontact result and new state.
        """
        return self._step(Action(
            action_type="recontact",
            patient_id=patient_id,
            hypothesis=hypothesis,
            confidence=confidence,
        ))

    def allocate_to_site(self, patient_id: str, site_id: str, hypothesis: str = "noise_dominant", confidence: float = 0.8) -> str:
        """Allocate a consented patient to a specific recruitment site for enrollment.

        This is the primary enrollment action. Only patients in allocation_candidates
        can be allocated.

        Args:
            patient_id: Patient ID from the allocation_candidates list.
            site_id: Site ID from the site_performance dictionary.
            hypothesis: Your hypothesis about which dynamic dominates.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing allocation result and new state.
        """
        return self._step(Action(
            action_type="allocate_to_site",
            patient_id=patient_id,
            site_id=site_id,
            hypothesis=hypothesis,
            confidence=confidence,
        ))

    def adjust_strategy(self, strategy_change: str, hypothesis: str = "noise_dominant", confidence: float = 0.6) -> str:
        """Adjust the recruitment strategy when no direct patient actions are available.

        Args:
            strategy_change: One of increase_outreach, relax_criteria, tighten_criteria, focus_site_A, focus_site_B, focus_site_C, negotiate_site_A, negotiate_site_B, negotiate_site_C.
            hypothesis: Your hypothesis about which dynamic dominates.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing strategy adjustment effect.
        """
        return self._step(Action(
            action_type="adjust_strategy",
            strategy_change=strategy_change,
            hypothesis=hypothesis,
            confidence=confidence,
        ))

    def plan_next_phase(self, target_phase: str, plan_summary: str = "advance the bottleneck") -> str:
        """Create or revise the current high-level recruitment plan.

        Args:
            target_phase: One of screening, conversion, allocation, retention, or recovery.
            plan_summary: Natural-language summary of the plan.

        Returns:
            Updated observation with plan feedback.
        """
        return self._step(Action(
            action_type="plan_next_phase",
            target_phase=target_phase,
            plan_summary=plan_summary,
        ))

    def summarize_and_index(self, memory_key: str, memory_payload: str) -> str:
        """Write a summary into indexed episodic memory for later retrieval.

        Args:
            memory_key: Key for the indexed memory item (e.g., step_20_progress).
            memory_payload: Summary text to store in memory.

        Returns:
            Updated observation confirming memory was stored.
        """
        return self._step(Action(
            action_type="summarize_and_index",
            memory_key=memory_key,
            memory_payload=memory_payload,
        ))

    def retrieve_relevant_history(self, memory_query: str) -> str:
        """Retrieve indexed memory entries relevant to the current situation.

        Args:
            memory_query: Query string describing what context is needed.

        Returns:
            Updated observation with retrieved memory context.
        """
        return self._step(Action(
            action_type="retrieve_relevant_history",
            memory_query=memory_query,
        ))

    def stop_recruitment(self) -> str:
        """End the current recruitment episode early.

        Only call this when you believe no further progress is possible.

        Returns:
            Final observation with episode results.
        """
        return self._step(Action(action_type="stop_recruitment"))

    def close(self) -> None:
        """Clean up resources. Called by TRL when the environment is no longer needed."""
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(self, action: Action) -> str:
        """Execute one environment step and return formatted observation."""
        if self.done:
            raise ValueError("Episode is finished. No more actions can be taken.")

        result = self._env.step(action)
        self.last_observation = result.observation.model_dump()
        self.reward = float(result.reward or 0)
        self.cumulative_reward += self.reward
        self.done = bool(result.done)

        # Track history for reward functions
        self.action_history.append(action.action_type)
        obs = self.last_observation
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0))
        if action.hypothesis:
            self.hypothesis_history.append(action.hypothesis)

        return self._format_observation()

    def _format_observation(self) -> str:
        """Format current observation as compact text for the model."""
        o = self.last_observation or {}
        avail = [p["id"] for p in o.get("available_patients", [])[:5]]
        recon = [p["id"] for p in o.get("recontact_candidates", [])[:5]]
        alloc = [p["id"] for p in o.get("allocation_candidates", [])[:5]]
        sites = list(o.get("site_performance", {}).keys())
        funnel = o.get("current_funnel", {})

        parts = [
            f"step={o.get('timestamp', 0)}",
            f"budget={o.get('budget_remaining', 0):.0f}",
            f"enrolled={o.get('enrolled_so_far', 0)}/{o.get('target_enrollment', 100)}",
        ]
        if avail:
            parts.append(f"available_patients={avail}")
        if recon:
            parts.append(f"recontact_candidates={recon}")
        if alloc:
            parts.append(f"allocation_candidates={alloc}")
        if sites:
            parts.append(f"sites={sites}")
        if funnel:
            parts.append(f"funnel={funnel}")

        events = o.get("recent_events", [])
        if events:
            parts.append(f"events={events[:3]}")

        return " ".join(parts)


# ------------------------------------------------------------------
# Reward functions for TRL GRPOTrainer
# ------------------------------------------------------------------

def reward_enrollment_progress(environments, **_) -> list[float]:
    """Fraction of target enrollment reached."""
    rewards = []
    for env in environments:
        obs = env.last_observation or {}
        enrolled = obs.get("enrolled_so_far", 0)
        target = max(1, obs.get("target_enrollment", 100))
        rewards.append(min(1.0, enrolled / target))
    return rewards


def reward_budget_efficiency(environments, **_) -> list[float]:
    """Enrollment per unit budget spent."""
    rewards = []
    for env in environments:
        obs = env.last_observation or {}
        ib = env.initial_budget or 1
        spent = max(0, ib - obs.get("budget_remaining", 0))
        target = max(1, obs.get("target_enrollment", 100))
        enrolled = obs.get("enrolled_so_far", 0)
        if spent > 0:
            rewards.append(min(1.0, (enrolled / target) / (spent / ib)))
        else:
            rewards.append(0.0)
    return rewards


def reward_screening_accuracy(environments, **_) -> list[float]:
    """Enrolled-to-screened ratio minus dropout penalty."""
    rewards = []
    for env in environments:
        funnel = (env.last_observation or {}).get("current_funnel", {})
        screened = funnel.get("screened", 0)
        if screened > 0:
            enrolled = funnel.get("enrolled", 0)
            dropped = funnel.get("dropped", 0)
            rewards.append(max(0, min(1, enrolled / screened - 0.5 * dropped / screened)))
        else:
            rewards.append(0.0)
    return rewards


def reward_action_diversity(environments, **_) -> list[float]:
    """Fraction of 8 possible action types used."""
    rewards = []
    for env in environments:
        if env.action_history:
            rewards.append(min(1.0, float(len(set(env.action_history))) / 8.0))
        else:
            rewards.append(0.0)
    return rewards


def reward_hypothesis_consistency(environments, **_) -> list[float]:
    """Penalizes erratic switching, rewards correct world-type match."""
    rewards = []
    for env in environments:
        hs = env.hypothesis_history
        if len(hs) < 2:
            rewards.append(0.5)
            continue
        switches = sum(1 for i in range(1, len(hs)) if hs[i] != hs[i - 1])
        consistency = 1.0 if switches <= 1 else max(0.2, 1 - (switches - 1) * 0.2)
        wt = (env.last_observation or {}).get("world_type", "")
        mapping = {"dropout_dominant": "dropout", "noise_dominant": "noise", "site_bias": "site_bias"}
        bonus = 0.2 if hs and mapping.get(hs[-1], "") == wt and wt else 0
        rewards.append(min(1.0, consistency * 0.8 + bonus))
    return rewards


# Convenience list for use in GRPOTrainer
REWARD_FUNCS = [
    reward_enrollment_progress,
    reward_budget_efficiency,
    reward_screening_accuracy,
    reward_action_diversity,
    reward_hypothesis_consistency,
]
