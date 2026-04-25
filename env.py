"""Core Adaptive Clinical Trial Recruitment environment implementing step/reset/state.

Simulates a 180-step clinical trial recruitment benchmark where each step
corresponds to one simulated day and agents optimize the
screening -> enrollment -> retention funnel under budget, time pressure,
site variability, and non-stationary patient quality.

Upgrades integrated:
  - Hypothesis channel: agent declares what it thinks is driving outcomes
  - Trajectory consistency: penalizes erratic hypothesis switching
  - Causal feedback: observation includes insight about dominant dynamics
  - Early commit pressure: step penalty + confidence calibration on finalize
  - Long-horizon mechanics: delayed effects, milestones, constraints, and site negotiation

Reward design (per step):
  - Screening success: +0.22 for successful screening (patient eligible)
  - Enrollment gain: +0.35 for new enrollment
  - Dropout penalty: -0.28 for each dropout event
  - Budget efficiency: 0.0 to +0.15 based on cost-effectiveness
  - Timeline bonus: 0.0 to +0.20 for being ahead of enrollment schedule
  - Curriculum bonus: +0.18 during easy-pool-reset periods if uncertainty < 0.3
  - Hypothesis bonus: +0.10 if hypothesis matches ground truth world_type
  - Consistency penalty: -0.10 if agent switches hypothesis > 2 times
  - Step pressure: -0.005 * (step/max_steps) (mild increasing urgency, max -0.005)
  - Confidence calibration: penalizes overconfident/underconfident finalize
"""

import copy
import random
from typing import Optional, Dict, Any, List

from models import Observation, Action, State, StepResult
from load_traces import (
    get_stage_horizon_days,
    get_task_trace,
    is_known_task,
    resolve_base_task_id,
)
from graders import GRADERS


class ClinicalRecruitmentEnv:
    """Clinical trial recruitment optimization environment.

    The agent manages the patient recruitment funnel:
    screening, site allocation, strategy adjustment, and retention
    across a 180-step trial recruitment window with one simulated day per step.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._task: Optional[str] = None
        self._trace: Dict[str, Any] = {}
        self._step: int = 0
        self._max_steps: int = 180
        self._done: bool = True
        self._total_reward: float = 0.0
        self._history: List[Dict[str, Any]] = []

        # Trial state
        self._budget_remaining: float = 0.0
        self._enrolled: int = 0
        self._target: int = 100
        self._deadline_days: int = 180

        # Patient pool (deep-copied from trace to allow mutation)
        self._patients: List[Dict[str, Any]] = []
        self._sites: Dict[str, Dict[str, float]] = {}
        self._events: Dict[int, List[str]] = {}
        self._curriculum: List[Dict[str, Any]] = []

        # Costs
        self._screening_cost: float = 800.0
        self._enrollment_cost: float = 1200.0
        self._recontact_cost: float = 100.0
        self._strategy_cost: float = 200.0

        # Rates
        self._dropout_base_rate: float = 0.10
        self._uncertainty_growth: float = 0.003
        self._uncertainty: float = 0.0

        # Funnel tracking
        self._funnel = {
            "contacted": 0,
            "screened": 0,
            "eligible": 0,
            "consented": 0,
            "enrolled": 0,
            "dropped": 0,
        }
        self._screening_backlog: int = 0
        self._recent_dropouts: List[
            int
        ] = []  # days when dropouts occurred (for 7-day rolling)

        # Strategy modifiers
        self._outreach_multiplier: float = 1.0
        self._criteria_strictness: float = (
            1.0  # 1.0 = normal, <1.0 = relaxed, >1.0 = tightened
        )
        self._site_focus: Optional[str] = None

        # RNG for deterministic simulation (seeded per task)
        self._rng: random.Random = random.Random(42)

        # Hypothesis tracking (Upgrade 1 + 2)
        self._hypothesis_history: List[str] = []
        self._last_hypothesis: str = "unknown"
        self._last_confidence: float = 0.5
        self._world_type: str = "unknown"

        # Long-horizon benchmark state
        self._delayed_effects: List[Dict[str, Any]] = []
        self._milestones: Dict[str, bool] = {}
        self._active_constraints: Dict[str, Any] = {}
        self._uncertainty_components: Dict[str, float] = {}
        self._patient_memory: Dict[str, int] = {}
        self._counterfactual_hint: str = ""
        self._current_plan: Dict[str, Any] = {}
        self._indexed_memory: Dict[str, Dict[str, Any]] = {}
        self._retrieved_memory_context: str = ""
        self._milestone_potential: float = 0.0
        self._active_milestone: str = ""
        self._hindsight_summary: Dict[str, Any] = {}
        self._site_negotiation_state: Dict[str, Dict[str, Any]] = {}
        self._oversight_queue: List[Dict[str, Any]] = []
        self._pareto_frontier: List[Dict[str, float]] = []
        self._token_budget_total: int = 12000
        self._token_budget_remaining: int = 12000
        self._token_usage_so_far: int = 0
        self._token_efficiency_score: float = 1.0

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
        if task is None:
            task = "easy_bench"
        if not is_known_task(task):
            raise ValueError(
                "Unknown task: "
                f"{task}. Available: ['easy_bench', 'medium_bench', 'hard_bench']"
            )

        self._task = task
        self._trace = get_task_trace(task)
        self._step = 0
        self._max_steps = int(self._trace.get("max_steps", self._trace.get("deadline_days", 180)))
        self._done = False
        self._total_reward = 0.0
        self._history = []

        # Default to deterministic per-task seeds, but honor explicit callers.
        if seed is None:
            seeds = {"easy_bench": 42, "medium_bench": 123, "hard_bench": 777}
            base_task = resolve_base_task_id(task)
            stage_horizon = get_stage_horizon_days(task)
            seed = seeds.get(base_task, 42)
            if stage_horizon is not None:
                seed += stage_horizon
        self._rng = random.Random(seed)

        # Deep-copy mutable state from trace
        self._patients = copy.deepcopy(self._trace["patients"])
        self._sites = copy.deepcopy(self._trace["sites"])
        self._events = copy.deepcopy(self._trace["events"])
        self._curriculum = copy.deepcopy(self._trace.get("curriculum", []))

        # Trial parameters
        self._budget_remaining = self._trace["budget"]
        self._target = self._trace["target_enrollment"]
        self._deadline_days = int(self._trace["deadline_days"])
        self._enrolled = 0

        # Costs
        self._screening_cost = self._trace["screening_cost"]
        self._enrollment_cost = self._trace["enrollment_cost"]
        self._recontact_cost = self._trace["recontact_cost"]
        self._strategy_cost = self._trace["strategy_cost"]

        # Rates
        self._dropout_base_rate = self._trace["dropout_base_rate"]
        self._uncertainty_growth = self._trace["uncertainty_growth"]
        self._uncertainty = 0.0

        # Reset funnel
        self._funnel = {
            "contacted": 0,
            "screened": 0,
            "eligible": 0,
            "consented": 0,
            "enrolled": 0,
            "dropped": 0,
        }
        self._screening_backlog = 0
        self._recent_dropouts = []

        # Reset strategy
        self._outreach_multiplier = 1.0
        self._criteria_strictness = 1.0
        self._site_focus = None

        # Reset hypothesis tracking
        self._hypothesis_history = []
        self._last_hypothesis = "unknown"
        self._last_confidence = 0.5
        self._world_type = self._trace.get("world_type", "unknown")

        for patient in self._patients:
            patient["priority_score"] = round(
                patient["eligibility_score"]
                * patient["willingness"]
                * (1.0 - patient["dropout_risk"]),
                3,
            )
            patient["followup_due_day"] = None
            patient["recontact_attempts"] = 0

        self._delayed_effects = []
        self._milestones = {
            "25pct": False,
            "50pct": False,
            "75pct": False,
            "100pct": False,
        }
        self._active_constraints = {
            "regulatory_hold_days": 0,
            "competitor_pressure": 0.0,
            "sentiment_pressure": 0.0,
            "protocol_version": 1,
            "sponsor_pressure": False,
            "backlog_pressure": False,
            "site_bottleneck": False,
            "focused_site": "",
        }
        self._uncertainty_components = {
            "patient_pool": 0.0,
            "site_operations": 0.0,
            "policy": 0.0,
        }
        self._patient_memory = {}
        self._counterfactual_hint = ""
        self._current_plan = {}
        self._indexed_memory = {}
        self._retrieved_memory_context = ""
        self._milestone_potential = 0.0
        self._active_milestone = ""
        self._hindsight_summary = {}
        self._site_negotiation_state = {}
        self._oversight_queue = []
        self._pareto_frontier = []
        self._token_budget_total = int(self._trace.get("token_budget", 12000))
        self._token_budget_remaining = self._token_budget_total
        self._token_usage_so_far = 0
        self._token_efficiency_score = 1.0
        self._refresh_long_horizon_state()

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"task": task, "difficulty": self._difficulty()},
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        outcome: Dict[str, Any] = {
            "screen_success": False,
            "enrolled": False,
            "dropout": False,
            "budget_efficiency": 0.0,
            "timeline_bonus": 0.0,
            "milestone_bonus": 0.0,
            "delayed_effects_triggered": 0,
            "plan_bonus": 0.0,
            "memory_bonus": 0.0,
            "milestone_potential_delta": 0.0,
            "plan_followthrough": False,
            "memory_hit": False,
            "token_cost": 0,
            "token_bonus": 0.0,
        }

        action_error = None
        potential_before = self._milestone_potential

        # -- Track hypothesis from agent (Upgrade 1) --
        hypothesis = action.hypothesis or "unknown"
        confidence = action.confidence
        self._hypothesis_history.append(hypothesis)
        self._last_hypothesis = hypothesis
        self._last_confidence = confidence
        token_cost = self._estimate_token_cost(action)
        self._token_usage_so_far += token_cost
        self._token_budget_remaining = max(0, self._token_budget_total - self._token_usage_so_far)
        outcome["token_cost"] = token_cost

        # -- Check curriculum injection (hard bench) --
        in_curriculum = False
        if self._curriculum:
            for inj in self._curriculum:
                if inj["start_day"] <= self._step <= inj["start_day"] + inj["duration"]:
                    in_curriculum = True
                    # During curriculum reset, uncertainty drops temporarily
                    self._uncertainty = max(0.0, self._uncertainty - 0.15)
                    break

        # -- Process day events --
        day_events = self._events.get(self._step, [])
        for evt in day_events:
            self._process_day_event(evt, outcome)

        # -- Trigger delayed consequences scheduled by earlier decisions --
        self._process_delayed_effects(outcome)

        # -- Apply action --
        if action.action_type == "screen_patient":
            outcome, action_error = self._do_screen(action, outcome)
        elif action.action_type == "recontact":
            outcome, action_error = self._do_recontact(action, outcome)
        elif action.action_type == "allocate_to_site":
            outcome, action_error = self._do_allocate(action, outcome)
        elif action.action_type == "adjust_strategy":
            outcome, action_error = self._do_strategy(action, outcome)
        elif action.action_type == "plan_next_phase":
            outcome, action_error = self._do_plan(action, outcome)
        elif action.action_type == "summarize_and_index":
            outcome, action_error = self._do_summarize_and_index(action, outcome)
        elif action.action_type == "retrieve_relevant_history":
            outcome, action_error = self._do_retrieve_relevant_history(action, outcome)
        elif action.action_type == "stop_recruitment":
            self._done = True

        # -- Natural progression: process screening backlog --
        if self._screening_backlog > 0:
            resolved = min(self._screening_backlog, 2)
            self._screening_backlog -= resolved

        # -- Natural dropout check for enrolled patients --
        dropout_rate = self._dropout_base_rate + self._uncertainty * 0.05
        enrolled_patients = [
            p for p in self._patients if p["enrolled"] and not p["dropped"]
        ]
        for p in enrolled_patients:
            if self._rng.random() < dropout_rate * p["dropout_risk"] * 0.1:
                p["dropped"] = True
                p["enrolled"] = False
                self._enrolled -= 1
                self._funnel["dropped"] += 1
                self._funnel["enrolled"] = max(0, self._funnel["enrolled"] - 1)
                self._recent_dropouts.append(self._step)
                outcome["dropout"] = True

        # -- Update uncertainty --
        pressure = self._active_constraints.get("competitor_pressure", 0.0)
        self._uncertainty = min(
            1.0, self._uncertainty + self._uncertainty_growth + pressure * 0.01
        )

        self._update_progress_milestones(outcome)
        self._refresh_long_horizon_state()
        outcome["milestone_potential_delta"] = round(
            self._milestone_potential - potential_before,
            4,
        )
        if self._current_plan and action.action_type not in {
            "plan_next_phase",
            "summarize_and_index",
            "retrieve_relevant_history",
            "stop_recruitment",
        }:
            outcome["plan_followthrough"] = self._action_matches_phase(
                action.action_type,
                self._current_plan.get("target_phase", ""),
                action.strategy_change,
            )
            if outcome["plan_followthrough"]:
                outcome["plan_bonus"] += 0.015
                self._current_plan["last_followthrough_step"] = self._step
            else:
                outcome["plan_bonus"] -= 0.005

        # -- Compute reward (Upgrade 4: includes hypothesis, consistency, commit pressure) --
        reward = self._compute_reward(
            outcome, in_curriculum, confidence, action.action_type, hypothesis
        )
        self._total_reward += reward

        # -- Record history --
        self._history.append(
            {
                "step": self._step,
                "action": action.action_type,
                "patient_id": action.patient_id,
                "site_id": action.site_id,
                "strategy": action.strategy_change,
                "hypothesis": hypothesis,
                "confidence": round(confidence, 3),
                "plan_id": action.plan_id,
                "plan_target_phase": action.target_phase,
                "memory_key": action.memory_key,
                "memory_query": action.memory_query,
                "enrolled": self._enrolled,
                "budget": round(self._budget_remaining, 2),
                "reward": round(reward, 4),
                "dropout": outcome["dropout"],
                "screen_success": outcome["screen_success"],
                "curriculum": in_curriculum,
                "milestone_bonus": round(outcome["milestone_bonus"], 3),
                "plan_bonus": round(outcome["plan_bonus"], 3),
                "memory_bonus": round(outcome["memory_bonus"], 3),
                "plan_followthrough": outcome["plan_followthrough"],
                "memory_hit": outcome["memory_hit"],
                "token_cost": outcome["token_cost"],
                "token_bonus": round(outcome["token_bonus"], 4),
                "token_efficiency_score": round(self._token_efficiency_score, 3),
                "milestone_potential": round(self._milestone_potential, 3),
                "milestone_potential_delta": round(
                    outcome["milestone_potential_delta"], 4
                ),
                "active_milestone": self._active_milestone,
                "delayed_effects_triggered": outcome["delayed_effects_triggered"],
                "error": action_error,
            }
        )

        self._advance_constraint_timers()

        self._step += 1
        self._done = (
            self._done or self._step >= self._max_steps or self._budget_remaining <= 0
        )

        # -- Check early success --
        if self._enrolled >= self._target:
            self._done = True

        self._refresh_long_horizon_state()
        if self._done:
            self._hindsight_summary = self._build_hindsight_summary()
        obs = self._make_observation()
        info: Dict[str, Any] = {
            "step": self._step,
            "reward_breakdown": {
                "total": round(reward, 4),
                "screen_success": outcome["screen_success"],
                "enrolled_new": outcome["enrolled"],
                "dropout": outcome["dropout"],
                "in_curriculum": in_curriculum,
                "milestone_bonus": round(outcome["milestone_bonus"], 3),
                "plan_bonus": round(outcome["plan_bonus"], 3),
                "memory_bonus": round(outcome["memory_bonus"], 3),
                "token_cost": outcome["token_cost"],
                "token_bonus": round(outcome["token_bonus"], 4),
                "milestone_potential_delta": round(
                    outcome["milestone_potential_delta"], 4
                ),
                "delayed_effects_triggered": outcome["delayed_effects_triggered"],
            },
            "last_action_error": action_error,
        }

        if self._done and self._hindsight_summary:
            info["hindsight_summary"] = self._hindsight_summary

        # If done, run grader
        if self._done:
            grader = GRADERS.get(resolve_base_task_id(self._task or ""))
            if grader:
                final_score = grader(obs, self._total_reward, self._history)
                # Strict clamp: must be in (0, 1), never exactly 0.0 or 1.0
                final_score = min(0.999, max(0.001, float(final_score)))
                info["final_score"] = final_score
                info["grader_task"] = resolve_base_task_id(self._task or "")

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        return State(
            task=self._task or "",
            step=self._step,
            max_steps=self._max_steps,
            time_to_deadline_days=max(0, self._deadline_days - self._step),
            done=self._done,
            enrolled_so_far=self._enrolled,
            target_enrollment=self._target,
            initial_budget=round(float(self._trace.get("budget", 0.0)), 2),
            budget_remaining=round(self._budget_remaining, 2),
            total_reward=round(self._total_reward, 4),
            history=self._history[-10:],
            milestones=dict(self._milestones),
            active_constraints=dict(self._active_constraints),
            delayed_effects_pending=len(self._delayed_effects),
            uncertainty_components=dict(self._uncertainty_components),
            current_plan=self._current_plan_snapshot(),
            indexed_memory_summary=self._summarize_indexed_memory(),
            retrieved_memory_context=self._retrieved_memory_context,
            milestone_potential=round(self._milestone_potential, 3),
            active_milestone=self._active_milestone,
            hindsight_summary=dict(self._hindsight_summary),
            token_budget_remaining=self._token_budget_remaining,
            token_usage_so_far=self._token_usage_so_far,
            token_efficiency_score=round(self._token_efficiency_score, 3),
        )

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history

    # ------- Action handlers -------

    def _do_screen(self, action: Action, outcome: dict) -> tuple:
        """Screen a patient: costs money, takes time, may find eligible candidate."""
        error = None
        if self._active_constraints.get("regulatory_hold_days", 0) > 0:
            self._screening_backlog += 1
            error = "regulatory_hold_active"
            return outcome, error
        if self._budget_remaining < self._screening_cost:
            error = "insufficient_budget"
            return outcome, error

        patient = self._find_patient(action.patient_id)
        if patient is None:
            error = "patient_not_found"
            return outcome, error
        if patient["screened"]:
            error = "already_screened"
            return outcome, error

        self._budget_remaining -= self._screening_cost
        patient["contacted"] = True
        patient["screened"] = True
        self._funnel["contacted"] += 1
        self._funnel["screened"] += 1

        # Screening outcome is probabilistic but deterministic (seeded RNG)
        outreach_bonus = 1.0 + (self._outreach_multiplier - 1.0) * 0.15
        adjusted_elig = min(
            0.99,
            (patient["eligibility_score"] * outreach_bonus) / self._criteria_strictness,
        )
        if self._rng.random() < adjusted_elig:
            patient["eligible"] = True
            self._funnel["eligible"] += 1
            outcome["screen_success"] = True

            # Auto-consent if eligible (simplified funnel)
            consent_prob = patient["willingness"] * (1.0 - self._uncertainty * 0.3)
            consent_prob *= min(1.2, 0.9 + self._outreach_multiplier * 0.1)
            consent_prob *= max(
                0.55,
                1.0
                - self._active_constraints.get("sentiment_pressure", 0.0)
                - self._active_constraints.get("competitor_pressure", 0.0) * 0.5,
            )
            consent_prob = max(0.05, min(0.95, consent_prob))
            if self._rng.random() < consent_prob:
                patient["consented"] = True
                patient["followup_due_day"] = self._step + 5
                self._funnel["consented"] += 1
                self._schedule_delayed_effect(
                    5,
                    {"type": "consent_window_closes", "patient_id": patient["id"]},
                )
            else:
                patient["followup_due_day"] = self._step + 3
                self._schedule_delayed_effect(
                    3,
                    {"type": "patient_cools_off", "patient_id": patient["id"]},
                )
        else:
            self._screening_backlog += 1  # failed screen adds to backlog

        self._recompute_patient_priority(patient)
        return outcome, error

    def _do_recontact(self, action: Action, outcome: dict) -> tuple:
        """Recontact a patient who was screened but not enrolled."""
        error = None
        if self._budget_remaining < self._recontact_cost:
            error = "insufficient_budget"
            return outcome, error

        patient = self._find_patient(action.patient_id)
        if patient is None:
            error = "patient_not_found"
            return outcome, error
        if patient["enrolled"]:
            error = "already_enrolled"
            return outcome, error
        if patient["dropped"]:
            error = "patient_dropped"
            return outcome, error

        self._budget_remaining -= self._recontact_cost
        patient["recontact_attempts"] = patient.get("recontact_attempts", 0) + 1

        pressure_factor = max(
            0.55,
            1.0
            - self._active_constraints.get("competitor_pressure", 0.0)
            - self._active_constraints.get("sentiment_pressure", 0.0) * 0.5,
        )
        due_bonus = 1.1 if patient.get("followup_due_day") is not None else 1.0

        # Recontact can re-engage dropped-interest patients
        if patient["consented"]:
            # Already consented, try to enroll directly
            enroll_prob = 0.6 * patient["willingness"] * (1 - patient["dropout_risk"])
            enroll_prob *= pressure_factor * due_bonus
            enroll_prob = max(0.05, min(0.95, enroll_prob))
            if (
                self._rng.random() < enroll_prob
                and self._budget_remaining >= self._enrollment_cost
            ):
                self._budget_remaining -= self._enrollment_cost
                patient["enrolled"] = True
                patient["followup_due_day"] = None
                self._enrolled += 1
                self._funnel["enrolled"] += 1
                outcome["enrolled"] = True
        elif patient["eligible"]:
            # Re-engage for consent
            consent_prob = patient["willingness"] * 0.5 * pressure_factor * due_bonus
            consent_prob = max(0.05, min(0.90, consent_prob))
            if self._rng.random() < consent_prob:
                patient["consented"] = True
                patient["followup_due_day"] = self._step + 4
                self._funnel["consented"] += 1
                self._schedule_delayed_effect(
                    4,
                    {"type": "consent_window_closes", "patient_id": patient["id"]},
                )
            else:
                patient["followup_due_day"] = self._step + 2

        self._recompute_patient_priority(patient)

        return outcome, error

    def _do_allocate(self, action: Action, outcome: dict) -> tuple:
        """Allocate a consented patient to a specific site for enrollment."""
        error = None
        patient = self._find_patient(action.patient_id)
        if patient is None:
            error = "patient_not_found"
            return outcome, error
        if not patient["consented"]:
            error = "patient_not_consented"
            return outcome, error
        if patient["enrolled"]:
            error = "already_enrolled"
            return outcome, error

        site_id = action.site_id
        if site_id is None or site_id not in self._sites:
            # Pick best available site
            site_id = self._pick_best_site()
            if site_id is None:
                error = "no_site_available"
                return outcome, error

        site = self._sites[site_id]
        if site["capacity_remaining"] <= 0:
            error = "site_at_capacity"
            return outcome, error

        if self._budget_remaining < self._enrollment_cost:
            error = "insufficient_budget"
            return outcome, error

        # Enrollment probability depends on site conversion rate and patient factors
        wait_factor = max(0.55, 1.0 - site["avg_wait_days"] / 20.0)
        retention_factor = 0.8 + site.get("retention_rate", 0.8) * 0.25
        enroll_prob = site["conversion_rate"] * (1.0 - patient["dropout_risk"] * 0.3)
        enroll_prob *= wait_factor * retention_factor
        if self._site_focus and self._site_focus == site_id:
            enroll_prob *= 1.15  # focused site gets bonus
        enroll_prob *= max(0.7, 1.0 - self._active_constraints.get("sentiment_pressure", 0.0))
        enroll_prob = max(0.05, min(0.98, enroll_prob))

        if self._rng.random() < enroll_prob:
            self._budget_remaining -= self._enrollment_cost
            patient["enrolled"] = True
            patient["site_assigned"] = site_id
            patient["followup_due_day"] = None
            site["capacity_remaining"] -= 1
            self._enrolled += 1
            self._funnel["enrolled"] += 1
            outcome["enrolled"] = True
        else:
            # Failed enrollment attempt — patient may drop
            if self._rng.random() < patient["dropout_risk"] * 0.3:
                patient["dropped"] = True
                self._funnel["dropped"] += 1
                outcome["dropout"] = True

        self._recompute_patient_priority(patient)
        return outcome, error

    def _do_strategy(self, action: Action, outcome: dict) -> tuple:
        """Adjust recruitment strategy."""
        error = None
        if self._budget_remaining < self._strategy_cost:
            error = "insufficient_budget"
            return outcome, error

        self._budget_remaining -= self._strategy_cost
        change = action.strategy_change or ""

        if change == "increase_outreach":
            self._outreach_multiplier = min(2.0, self._outreach_multiplier + 0.2)
            self._schedule_delayed_effect(2, {"type": "outreach_wave", "strength": 0.06})
        elif change == "relax_criteria":
            self._criteria_strictness = max(0.6, self._criteria_strictness - 0.15)
            self._schedule_delayed_effect(
                4,
                {"type": "criteria_relax_tail_risk", "amount": 0.05},
            )
        elif change == "tighten_criteria":
            self._criteria_strictness = min(1.5, self._criteria_strictness + 0.15)
            self._schedule_delayed_effect(
                3,
                {"type": "criteria_tighten_stability", "amount": 0.04},
            )
        elif change.startswith("focus_site_"):
            site_key = "site_" + change.replace("focus_site_", "")
            if site_key in self._sites:
                self._site_focus = site_key
            else:
                error = "site_not_found"
        elif change.startswith("negotiate_site_"):
            site_key = "site_" + change.replace("negotiate_site_", "")
            if site_key in self._sites:
                self._site_negotiation_state.setdefault(site_key, {}).update(
                    {
                        "offers_made": int(
                            self._site_negotiation_state.get(site_key, {}).get(
                                "offers_made", 0
                            )
                        )
                        + 1,
                        "last_offer_step": self._step,
                    }
                )
                self._schedule_delayed_effect(
                    2,
                    {"type": "site_negotiation", "site_id": site_key},
                )
            else:
                error = "site_not_found"
        else:
            error = "unknown_strategy"

        return outcome, error

    def _do_plan(self, action: Action, outcome: dict) -> tuple:
        """Create or refresh an explicit high-level plan for the next phase."""
        error = None
        valid_phases = {"screening", "conversion", "allocation", "retention", "recovery"}
        target_phase = (action.target_phase or "").strip().lower()
        if target_phase not in valid_phases:
            return outcome, "invalid_target_phase"

        summary = (action.plan_summary or "").strip() or self._default_plan_summary(
            target_phase
        )
        plan_id = (action.plan_id or f"plan-{self._step}-{target_phase}").strip()
        self._current_plan = {
            "plan_id": plan_id,
            "target_phase": target_phase,
            "summary": summary[:180],
            "created_step": self._step,
            "last_followthrough_step": None,
            "autogenerated": False,
        }
        outcome["plan_bonus"] += 0.005
        return outcome, error

    def _do_summarize_and_index(self, action: Action, outcome: dict) -> tuple:
        """Write a compact episodic memory entry into an indexed store."""
        error = None
        key = (action.memory_key or "").strip().lower()
        payload = (action.memory_payload or "").strip()
        if not key:
            return outcome, "missing_memory_key"
        if not payload:
            payload = self._default_memory_payload(key)

        tags = self._infer_memory_tags(key, payload)
        self._indexed_memory[key] = {
            "key": key,
            "summary": payload[:220],
            "tags": tags,
            "step": self._step,
        }
        return outcome, error

    def _do_retrieve_relevant_history(self, action: Action, outcome: dict) -> tuple:
        """Retrieve indexed memory entries relevant to the current phase or query."""
        error = None
        query = (action.memory_query or "").strip().lower()
        if not query:
            return outcome, "missing_memory_query"

        matches = self._retrieve_indexed_memory(query)
        if matches:
            self._retrieved_memory_context = " | ".join(matches[:2])[:280]
            outcome["memory_bonus"] += 0.005
            outcome["memory_hit"] = True
        else:
            self._retrieved_memory_context = (
                f"No indexed history matched '{query}' at step {self._step}."
            )
            outcome["memory_bonus"] -= 0.01
        return outcome, error

    # ------- Helper methods -------

    def _find_patient(self, patient_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if patient_id is None:
            return None  # Agent must explicitly select a patient
        for p in self._patients:
            if p["id"] == patient_id:
                return p
        return None

    def _pick_best_site(self) -> Optional[str]:
        best_id = None
        best_score = -1
        for sid, site in self._sites.items():
            if site["capacity_remaining"] > 0:
                capacity_ratio = site["capacity_remaining"] / max(1, site["capacity_total"])
                score = site["conversion_rate"] * site.get("retention_rate", 0.8)
                score *= (0.5 + capacity_ratio)
                score /= max(1.0, site["avg_wait_days"])
                if self._site_focus == sid:
                    score *= 1.1
                if score > best_score:
                    best_score = score
                    best_id = sid
        return best_id

    def _default_plan_summary(self, target_phase: str) -> str:
        defaults = {
            "screening": "Screen high-priority unscreened patients while uncertainty stays manageable.",
            "conversion": "Recontact eligible candidates before they cool off and convert consent faster.",
            "allocation": "Allocate consented patients to the strongest available site before delay risk rises.",
            "retention": "Protect enrolled patients with high dropout risk and avoid fragile conversions.",
            "recovery": "Recover from active constraints and rebuild pipeline stability before scaling volume.",
        }
        return defaults.get(target_phase, "Advance the next long-horizon recruitment phase.")

    def _default_memory_payload(self, key: str) -> str:
        if "site" in key:
            site_id = self._pick_best_site() or "site_unknown"
            site = self._sites.get(site_id, {})
            return (
                f"{site_id} snapshot: conv={site.get('conversion_rate', 0):.2f}, "
                f"wait={site.get('avg_wait_days', 0):.1f}, "
                f"capacity={site.get('capacity_remaining', 0)}"
            )
        if "retention" in key or "dropout" in key:
            return (
                f"Retention snapshot: at_risk_enrolled={self._patient_memory.get('at_risk_enrolled', 0)}, "
                f"dropout_7d={self._rolling_dropout_rate():.2f}"
            )
        return (
            f"Pipeline snapshot: followup_due={self._patient_memory.get('followup_due', 0)}, "
            f"consented_pending={self._patient_memory.get('consented_pending_allocation', 0)}, "
            f"active_milestone={self._active_milestone or 'none'}"
        )

    def _infer_memory_tags(self, key: str, payload: str) -> List[str]:
        text = f"{key} {payload}".lower()
        tags = []
        for tag in ("screening", "conversion", "allocation", "retention", "recovery", "site", "dropout", "milestone"):
            if tag in text:
                tags.append(tag)
        if not tags:
            tags.append("general")
        return tags

    def _retrieve_indexed_memory(self, query: str) -> List[str]:
        query_terms = {term for term in query.lower().replace("_", " ").split() if term}
        ranked: List[tuple] = []
        for entry in self._indexed_memory.values():
            haystack = f"{entry.get('key', '')} {entry.get('summary', '')} {' '.join(entry.get('tags', []))}".lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            if overlap <= 0:
                continue
            ranked.append((overlap, int(entry.get("step", 0)), entry.get("summary", "")))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [summary for _, _, summary in ranked]

    def _current_plan_snapshot(self) -> Dict[str, Any]:
        if not self._current_plan:
            return {}
        return {
            "plan_id": self._current_plan.get("plan_id", ""),
            "target_phase": self._current_plan.get("target_phase", ""),
            "summary": self._current_plan.get("summary", ""),
            "created_step": self._current_plan.get("created_step"),
            "last_followthrough_step": self._current_plan.get("last_followthrough_step"),
        }

    def _estimate_token_cost(self, action: Action) -> int:
        if action.token_cost is not None:
            return max(0, int(action.token_cost))
        base_costs = {
            "screen_patient": 18,
            "recontact": 20,
            "allocate_to_site": 22,
            "adjust_strategy": 26,
            "plan_next_phase": 55,
            "summarize_and_index": 42,
            "retrieve_relevant_history": 36,
            "stop_recruitment": 12,
        }
        return base_costs.get(action.action_type, 20)

    def _compute_token_efficiency(self) -> float:
        progress = self._enrolled / max(1, self._target)
        usage_ratio = self._token_usage_so_far / max(1, self._token_budget_total)
        plan_followthrough = sum(1 for item in self._history if item.get("plan_followthrough"))
        memory_hits = sum(1 for item in self._history if item.get("memory_hit"))
        quality_bonus = min(0.18, plan_followthrough * 0.01 + memory_hits * 0.015)
        score = 1.0 - usage_ratio * 0.65 + progress * 0.25 + quality_bonus
        return round(max(0.0, min(1.0, score)), 3)

    def _summarize_indexed_memory(self) -> Dict[str, int]:
        summary: Dict[str, int] = {"entries": len(self._indexed_memory)}
        for entry in self._indexed_memory.values():
            for tag in entry.get("tags", []):
                summary[tag] = summary.get(tag, 0) + 1
        return summary

    def _action_matches_phase(
        self,
        action_type: str,
        target_phase: str,
        strategy_change: Optional[str],
    ) -> bool:
        mapping = {
            "screening": {"screen_patient"},
            "conversion": {"recontact"},
            "allocation": {"allocate_to_site"},
            "retention": {"recontact"},
            "recovery": {"adjust_strategy"},
        }
        if target_phase == "recovery" and strategy_change:
            return str(strategy_change).startswith(("tighten_criteria", "negotiate_site_", "focus_site_", "increase_outreach"))
        return action_type in mapping.get(target_phase, set())

    def _process_random_dropout(self) -> bool:
        enrolled_patients = [
            p for p in self._patients if p["enrolled"] and not p["dropped"]
        ]
        if enrolled_patients:
            victim = self._rng.choice(enrolled_patients)
            victim["dropped"] = True
            victim["enrolled"] = False
            self._enrolled -= 1
            self._funnel["dropped"] += 1
            self._funnel["enrolled"] = max(0, self._funnel["enrolled"] - 1)
            self._recent_dropouts.append(self._step)
            return True
        return False

    def _process_site_delay(self) -> Optional[str]:
        site_ids = list(self._sites.keys())
        if site_ids:
            affected = self._rng.choice(site_ids)
            self._sites[affected]["avg_wait_days"] += 2.0
            return affected
        return None

    def _reduce_random_site_capacity(self) -> Optional[str]:
        site_ids = list(self._sites.keys())
        if site_ids:
            affected = self._rng.choice(site_ids)
            self._sites[affected]["capacity_remaining"] = max(
                0, self._sites[affected]["capacity_remaining"] - 3
            )
            return affected
        return None

    def _schedule_delayed_effect(self, delay_days: int, effect: Dict[str, Any]):
        payload = dict(effect)
        payload["due_step"] = self._step + max(1, int(delay_days))
        self._delayed_effects.append(payload)

    def _process_day_event(self, evt: str, outcome: Dict[str, Any]):
        if evt == "patient_dropout":
            outcome["dropout"] = self._process_random_dropout() or outcome["dropout"]
        elif evt == "site_delay":
            affected = self._process_site_delay()
            if affected:
                self._schedule_delayed_effect(
                    4,
                    {"type": "site_recovery", "site_id": affected},
                )
        elif evt == "regulatory_hold":
            self._screening_backlog += 3
            self._active_constraints["regulatory_hold_days"] = max(
                int(self._active_constraints.get("regulatory_hold_days", 0)), 2
            )
        elif evt == "site_audit":
            affected = self._process_site_delay()
            if affected:
                self._active_constraints["site_bottleneck"] = True
                self._schedule_delayed_effect(5, {"type": "site_recovery", "site_id": affected})
        elif evt == "irb_delay":
            self._active_constraints["regulatory_hold_days"] = max(
                int(self._active_constraints.get("regulatory_hold_days", 0)), 3
            )
        elif evt == "consent_form_revision":
            self._schedule_delayed_effect(2, {"type": "consent_form_revision"})
        elif evt == "site_capacity_reduced":
            affected = self._reduce_random_site_capacity()
            if affected:
                self._schedule_delayed_effect(
                    6,
                    {"type": "capacity_recovery", "site_id": affected},
                )
        elif evt == "seasonal_slowdown":
            self._outreach_multiplier = max(0.5, self._outreach_multiplier - 0.1)
            self._active_constraints["sentiment_pressure"] = min(
                0.4,
                float(self._active_constraints.get("sentiment_pressure", 0.0)) + 0.03,
            )
        elif evt == "new_competitor_trial":
            self._schedule_delayed_effect(
                3,
                {"type": "competitor_pressure", "amount": 0.12},
            )
        elif evt == "patient_complaint":
            self._schedule_delayed_effect(
                2,
                {"type": "sentiment_pressure", "amount": 0.08},
            )
        elif evt == "screening_backlog":
            self._screening_backlog += 2
        elif evt == "protocol_amendment":
            self._schedule_delayed_effect(1, {"type": "protocol_shift"})

    def _process_delayed_effects(self, outcome: Dict[str, Any]):
        pending: List[Dict[str, Any]] = []
        for effect in self._delayed_effects:
            if effect.get("due_step", self._step + 1) <= self._step:
                self._apply_delayed_effect(effect, outcome)
                outcome["delayed_effects_triggered"] += 1
            else:
                pending.append(effect)
        self._delayed_effects = pending

    def _apply_delayed_effect(self, effect: Dict[str, Any], outcome: Dict[str, Any]):
        effect_type = effect.get("type", "")

        if effect_type == "patient_cools_off":
            patient = self._find_patient(effect.get("patient_id"))
            if (
                patient
                and patient["eligible"]
                and not patient["consented"]
                and not patient["enrolled"]
                and patient.get("followup_due_day") is not None
                and patient["followup_due_day"] <= self._step
            ):
                patient["willingness"] = round(max(0.1, patient["willingness"] - 0.12), 3)
                patient["followup_due_day"] = None
                self._recompute_patient_priority(patient)
        elif effect_type == "consent_window_closes":
            patient = self._find_patient(effect.get("patient_id"))
            if (
                patient
                and patient["consented"]
                and not patient["enrolled"]
                and patient.get("followup_due_day") is not None
                and patient["followup_due_day"] <= self._step
            ):
                patient["consented"] = False
                patient["followup_due_day"] = self._step + 2
                self._funnel["consented"] = max(0, self._funnel["consented"] - 1)
                self._schedule_delayed_effect(
                    2,
                    {"type": "patient_cools_off", "patient_id": patient["id"]},
                )
                self._recompute_patient_priority(patient)
        elif effect_type == "competitor_pressure":
            self._active_constraints["competitor_pressure"] = round(
                min(
                    0.6,
                    float(self._active_constraints.get("competitor_pressure", 0.0))
                    + float(effect.get("amount", 0.0)),
                ),
                3,
            )
        elif effect_type == "sentiment_pressure":
            self._active_constraints["sentiment_pressure"] = round(
                min(
                    0.4,
                    float(self._active_constraints.get("sentiment_pressure", 0.0))
                    + float(effect.get("amount", 0.0)),
                ),
                3,
            )
        elif effect_type == "protocol_shift":
            self._active_constraints["protocol_version"] = int(
                self._active_constraints.get("protocol_version", 1)
            ) + 1
            self._criteria_strictness = min(1.4, self._criteria_strictness + 0.05)
            self._screening_backlog += 1
            self._uncertainty = min(1.0, self._uncertainty + 0.04)
            self._active_constraints["schema_delta"] = "eligibility tightened"
        elif effect_type == "consent_form_revision":
            self._active_constraints["sentiment_pressure"] = round(
                min(0.4, float(self._active_constraints.get("sentiment_pressure", 0.0)) + 0.04),
                3,
            )
        elif effect_type == "site_negotiation":
            site_id = effect.get("site_id")
            site = self._sites.get(site_id or "")
            if site:
                state = self._site_negotiation_state.setdefault(site_id or "", {})
                state["counteroffers"] = int(state.get("counteroffers", 0)) + 1
                site["conversion_rate"] = round(
                    min(0.98, site["conversion_rate"] + 0.04),
                    3,
                )
                site["avg_wait_days"] = round(max(1.0, site["avg_wait_days"] - 1.0), 1)
                site["capacity_remaining"] = min(
                    site["capacity_total"],
                    max(
                        site["capacity_remaining"] + 2,
                        int(state.get("private_capacity_floor", 1)),
                    ),
                )
        elif effect_type == "criteria_relax_tail_risk":
            self._uncertainty = min(
                1.0,
                self._uncertainty + float(effect.get("amount", 0.0)),
            )
        elif effect_type == "criteria_tighten_stability":
            self._uncertainty = max(
                0.0,
                self._uncertainty - float(effect.get("amount", 0.0)),
            )
        elif effect_type == "outreach_wave":
            strength = float(effect.get("strength", 0.05))
            candidates = [
                p for p in self._patients if not p["screened"] and not p["dropped"]
            ]
            candidates.sort(key=lambda p: p.get("priority_score", 0.0), reverse=True)
            for patient in candidates[:3]:
                patient["willingness"] = round(
                    min(0.95, patient["willingness"] + strength),
                    3,
                )
                self._recompute_patient_priority(patient)
        elif effect_type == "site_recovery":
            site_id = effect.get("site_id")
            site = self._sites.get(site_id or "")
            if site:
                site["avg_wait_days"] = round(max(1.0, site["avg_wait_days"] - 1.0), 1)
        elif effect_type == "capacity_recovery":
            site_id = effect.get("site_id")
            site = self._sites.get(site_id or "")
            if site:
                site["capacity_remaining"] = min(
                    site["capacity_total"],
                    site["capacity_remaining"] + 1,
                )

    def _update_progress_milestones(self, outcome: Dict[str, Any]):
        progress = self._enrolled / max(1, self._target)
        for label, threshold, bonus in [
            ("25pct", 0.25, 0.02),
            ("50pct", 0.50, 0.03),
            ("75pct", 0.75, 0.04),
            ("100pct", 1.00, 0.06),
        ]:
            if progress >= threshold and not self._milestones.get(label, False):
                self._milestones[label] = True
                outcome["milestone_bonus"] += bonus

    def _advance_constraint_timers(self):
        if self._active_constraints.get("regulatory_hold_days", 0) > 0:
            self._active_constraints["regulatory_hold_days"] -= 1
        for key in ("competitor_pressure", "sentiment_pressure"):
            self._active_constraints[key] = round(
                max(0.0, float(self._active_constraints.get(key, 0.0)) - 0.01),
                3,
            )

    def _refresh_long_horizon_state(self):
        avg_wait = 0.0
        low_capacity_sites = 0
        if self._sites:
            avg_wait = sum(site["avg_wait_days"] for site in self._sites.values()) / len(
                self._sites
            )
            low_capacity_sites = sum(
                1
                for site in self._sites.values()
                if site["capacity_remaining"] <= max(2, int(site["capacity_total"] * 0.1))
            )

        expected_progress = (self._step + 1) / max(1, self._max_steps)
        actual_progress = self._enrolled / max(1, self._target)
        self._active_constraints["backlog_pressure"] = self._screening_backlog >= 4
        self._active_constraints["site_bottleneck"] = (
            low_capacity_sites > 0 or avg_wait >= 7.0
        )
        self._active_constraints["sponsor_pressure"] = actual_progress + 0.05 < expected_progress
        self._active_constraints["focused_site"] = self._site_focus or ""

        self._uncertainty_components = {
            "patient_pool": round(
                min(
                    1.0,
                    self._uncertainty
                    + float(self._active_constraints.get("competitor_pressure", 0.0))
                    * 0.4,
                ),
                3,
            ),
            "site_operations": round(
                min(
                    1.0,
                    (avg_wait / 15.0)
                    + (low_capacity_sites / max(1, len(self._sites))) * 0.5,
                ),
                3,
            ),
            "policy": round(
                min(
                    1.0,
                    (max(0, int(self._active_constraints.get("protocol_version", 1)) - 1)
                    * 0.1)
                    + int(self._active_constraints.get("regulatory_hold_days", 0)) * 0.15
                    + float(self._active_constraints.get("sentiment_pressure", 0.0)),
                ),
                3,
            ),
        }
        self._patient_memory = self._summarize_patient_memory()
        self._active_milestone = self._compute_active_milestone()
        self._milestone_potential = self._compute_milestone_potential()
        self._trim_or_seed_plan()
        self._token_efficiency_score = self._compute_token_efficiency()
        self._update_oversight_and_frontier()
        self._counterfactual_hint = self._build_counterfactual_hint()

    def _update_oversight_and_frontier(self):
        point = {
            "enrolled": float(self._enrolled),
            "budget": float(self._budget_remaining),
            "token_efficiency": float(self._token_efficiency_score),
        }
        self._pareto_frontier.append(point)
        self._pareto_frontier.sort(
            key=lambda item: (item["enrolled"], item["budget"], item["token_efficiency"]),
            reverse=True,
        )
        self._pareto_frontier = self._pareto_frontier[:10]
        if self._active_constraints.get("regulatory_hold_days", 0) > 0 or self._screening_backlog >= 4:
            self._oversight_queue.append(
                {
                    "step": self._step,
                    "reason": "regulatory_or_backlog_pressure",
                }
            )
            self._oversight_queue = self._oversight_queue[-8:]

    def _compute_active_milestone(self) -> str:
        progress = self._enrolled / max(1, self._target)
        for label, threshold in [
            ("25pct", 0.25),
            ("50pct", 0.50),
            ("75pct", 0.75),
            ("100pct", 1.00),
        ]:
            if progress < threshold:
                return label
        return "complete"

    def _compute_milestone_potential(self) -> float:
        progress = self._enrolled / max(1, self._target)
        consented = self._patient_memory.get("consented_pending_allocation", 0)
        followup_due = self._patient_memory.get("followup_due", 0)
        high_priority = self._patient_memory.get("high_priority_candidates", 0)
        penalties = (
            int(self._active_constraints.get("regulatory_hold_days", 0)) * 0.05
            + float(self._active_constraints.get("competitor_pressure", 0.0)) * 0.10
            + float(self._active_constraints.get("sentiment_pressure", 0.0)) * 0.08
            + min(0.15, self._screening_backlog * 0.01)
        )
        readiness = min(
            0.55,
            consented * 0.03 + followup_due * 0.02 + high_priority * 0.015,
        )
        pace_bonus = min(0.20, max(0.0, progress - (self._step / max(1, self._max_steps))) * 0.5)
        potential = 0.20 + progress * 0.35 + readiness + pace_bonus - penalties
        return round(max(0.0, min(1.0, potential)), 3)

    def _trim_or_seed_plan(self):
        if self._current_plan:
            created = int(self._current_plan.get("created_step", self._step))
            target_phase = self._current_plan.get("target_phase", "")
            autogenerated = bool(self._current_plan.get("autogenerated", False))
            if self._active_milestone == "complete":
                self._current_plan = {}
                return
            stale = self._step - created > (18 if autogenerated else 24)
            frontier_mismatch = autogenerated and not self._phase_matches_frontier(
                target_phase
            )
            if stale or frontier_mismatch:
                self._current_plan = {}

        if not self._current_plan and self._active_milestone != "complete":
            target_phase = self._recommend_phase_for_frontier()
            self._current_plan = {
                "plan_id": f"autoplan-{self._step}-{target_phase}",
                "target_phase": target_phase,
                "summary": self._default_plan_summary(target_phase),
                "created_step": self._step,
                "last_followthrough_step": None,
                "autogenerated": True,
            }

    def _phase_matches_frontier(self, target_phase: str) -> bool:
        if self._active_milestone == "complete":
            return True
        return target_phase == self._recommend_phase_for_frontier()

    def _recommend_phase_for_frontier(self) -> str:
        if self._active_constraints.get("regulatory_hold_days", 0) > 0:
            return "recovery"
        if self._patient_memory.get("consented_pending_allocation", 0) > 0:
            return "allocation"
        if self._patient_memory.get("followup_due", 0) > 0 or self._patient_memory.get(
            "eligible_pending_consent", 0
        ) > 0:
            return "conversion"
        if self._patient_memory.get("at_risk_enrolled", 0) > 0 and self._task == "hard_bench":
            return "retention"
        return "screening"

    def _summarize_patient_memory(self) -> Dict[str, int]:
        return {
            "high_priority_candidates": sum(
                1
                for p in self._patients
                if not p["screened"]
                and not p["dropped"]
                and p.get("priority_score", 0.0) >= 0.35
            ),
            "eligible_pending_consent": sum(
                1
                for p in self._patients
                if p["eligible"] and not p["consented"] and not p["enrolled"] and not p["dropped"]
            ),
            "consented_pending_allocation": sum(
                1
                for p in self._patients
                if p["consented"] and not p["enrolled"] and not p["dropped"]
            ),
            "at_risk_enrolled": sum(
                1
                for p in self._patients
                if p["enrolled"] and p["dropout_risk"] >= 0.35 and not p["dropped"]
            ),
            "followup_due": sum(
                1
                for p in self._patients
                if p.get("followup_due_day") is not None
                and p["followup_due_day"] <= self._step + 2
                and not p["dropped"]
                and not p["enrolled"]
            ),
            "dropped": sum(1 for p in self._patients if p["dropped"]),
            "high_recontact_pressure": sum(
                1
                for p in self._patients
                if p.get("recontact_attempts", 0) >= 2 and not p["enrolled"] and not p["dropped"]
            ),
            "oversight_alerts": len(self._oversight_queue),
        }

    def _build_counterfactual_hint(self) -> str:
        if self._active_constraints.get("regulatory_hold_days", 0) > 0:
            return (
                "Counterfactual: while screening is blocked, recontact eligible patients "
                "or negotiate a site."
            )
        if self._patient_memory.get("followup_due", 0) > 0:
            return (
                "Counterfactual: recontact eligible patients before their follow-up "
                "window closes."
            )
        if self._patient_memory.get("consented_pending_allocation", 0) > 0:
            best_site = self._pick_best_site()
            if best_site:
                return (
                    f"Counterfactual: allocate consented patients to {best_site} "
                    "before they cool off."
                )
        if self._active_constraints.get("site_bottleneck", False):
            return (
                "Counterfactual: negotiate a congested site or shift focus to a lower-wait site."
            )
        if self._uncertainty_components.get("patient_pool", 0.0) > 0.4:
            return "Counterfactual: tighten criteria briefly to trade volume for stability."
        if self._pareto_frontier:
            best = self._pareto_frontier[0]
            return (
                "Counterfactual: target a higher-efficiency frontier point with "
                f"enrolled={int(best['enrolled'])} and budget={best['budget']:.0f}."
            )
        return "Counterfactual: keep screening high-priority patients while budget remains."

    def _counterfactual_rollout_summary(self) -> Dict[str, float]:
        next_enroll_if_allocate = min(
            1.0,
            self._patient_memory.get("consented_pending_allocation", 0) * 0.35,
        )
        next_followup_if_recontact = min(
            1.0,
            self._patient_memory.get("followup_due", 0) * 0.25,
        )
        return {
            "allocate_gain_estimate": round(next_enroll_if_allocate, 3),
            "recontact_gain_estimate": round(next_followup_if_recontact, 3),
        }

    def _rolling_dropout_rate(self) -> float:
        recent_drops = [d for d in self._recent_dropouts if d >= self._step - 7]
        enrolled_count = max(1, self._enrolled)
        return min(1.0, len(recent_drops) / enrolled_count) if enrolled_count > 0 else 0.0

    def _build_hindsight_summary(self) -> Dict[str, Any]:
        action_counts: Dict[str, int] = {}
        phase_followthrough = 0
        memory_hits = 0
        for item in self._history:
            action = item.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
            if item.get("plan_followthrough"):
                phase_followthrough += 1
            if item.get("memory_hit"):
                memory_hits += 1

        strongest_action = max(action_counts, key=action_counts.get) if action_counts else "none"
        weakest_milestone = self._active_milestone if self._active_milestone else "complete"
        recovery_suggestion = self._build_counterfactual_hint().replace("Counterfactual: ", "")
        return {
            "dominant_action": strongest_action,
            "phase_followthrough_steps": phase_followthrough,
            "memory_hits": memory_hits,
            "milestone_potential_final": round(self._milestone_potential, 3),
            "token_efficiency_score": round(self._token_efficiency_score, 3),
            "token_usage_so_far": self._token_usage_so_far,
            "weakest_milestone": weakest_milestone,
            "recovery_suggestion": recovery_suggestion,
            "score_proxy": round(
                max(
                    0.0,
                    min(
                        1.0,
                        (self._enrolled / max(1, self._target)) * 0.6
                        + phase_followthrough * 0.01
                        + memory_hits * 0.01,
                    ),
                ),
                3,
            ),
            "epistemic": round(min(1.0, self._uncertainty * 0.55 + self._screening_backlog * 0.015), 3),
            "aleatoric": round(min(1.0, self._uncertainty * 0.45 + self._rolling_dropout_rate() * 0.4), 3),
        }

    def _recompute_patient_priority(self, patient: Dict[str, Any]):
        patient["priority_score"] = round(
            patient["eligibility_score"]
            * patient["willingness"]
            * (1.0 - patient["dropout_risk"]),
            3,
        )

    def _consistency_penalty(self) -> float:
        """Penalize erratic hypothesis switching, but cap the total penalty.
        Old behavior: 0.10 every step after >2 switches = unbounded (-18.0 over 180 steps).
        New behavior: penalty proportional to switch count, capped at 0.05 per step max.
        """
        if len(self._hypothesis_history) < 2:
            return 0.0
        switches = 0
        for i in range(1, len(self._hypothesis_history)):
            if self._hypothesis_history[i] != self._hypothesis_history[i - 1]:
                switches += 1
        if switches <= 2:
            return 0.0
        # Mild penalty that doesn't grow unboundedly
        return min(0.05, (switches - 2) * 0.01)

    def _hypothesis_bonus(self, hypothesis: str) -> float:
        """Upgrade 1: Reward agent for correct hypothesis about world dynamics."""
        mapping = {
            "noise_dominant": "noise",
            "dropout_dominant": "dropout",
            "site_bias": "site_bias",
        }
        if mapping.get(hypothesis, "") == self._world_type:
            return 0.10
        return 0.0

    def _compute_reward(
        self,
        outcome: dict,
        in_curriculum: bool,
        confidence: float,
        action_type: str,
        hypothesis: str,
    ) -> float:
        """Compute per-step reward with partial credit + upgrades.

        Includes: hypothesis bonus, consistency penalty, step pressure,
        confidence calibration on finalize.
        """
        r = 0.0

        # Screening success
        if outcome.get("screen_success"):
            r += 0.30

        # Enrollment gain (primary goal - high signal)
        if outcome.get("enrolled"):
            r += 0.50

        # Dropout penalty
        if outcome.get("dropout"):
            r -= 0.35

        # Budget efficiency: small reward for not wasting money
        if self._budget_remaining > 0 and self._trace.get("budget", 1) > 0:
            budget_ratio = self._budget_remaining / self._trace["budget"]
            r += budget_ratio * 0.05

        # Timeline bonus: reward for being ahead of enrollment schedule
        if self._target > 0 and self._max_steps > 0:
            expected_progress = (self._step + 1) / self._max_steps
            actual_progress = self._enrolled / self._target
            if actual_progress > expected_progress:
                r += min(0.25, (actual_progress - expected_progress) * 0.50)

        # Curriculum bonus (hard bench)
        if in_curriculum and self._uncertainty < 0.3:
            r += 0.18

        # Milestones reward sustained progress instead of only terminal success.
        r += outcome.get("milestone_bonus", 0.0)

        # Base reward: minimal to avoid rewarding inaction
        r += 0.01

        # --- Inaction penalty (FIX 4) ---
        # Penalties are mild so that planning/memory actions + their bonuses
        # can break even or go slightly positive when used well.
        if action_type in ("screen_patient", "recontact", "allocate_to_site"):
            # Productive actions: no penalty
            pass
        elif action_type == "adjust_strategy":
            # Strategy is fine but not productive by itself
            r -= 0.02
        elif action_type == "plan_next_phase":
            r -= 0.03  # was -0.07, reduced so plan_bonus can offset
        elif action_type in ("summarize_and_index", "retrieve_relevant_history"):
            r -= 0.03  # was -0.06, reduced so memory_bonus can offset
        elif action_type == "stop_recruitment":
            # Handled below in confidence calibration
            pass
        else:
            # Unknown action
            r -= 0.05

        r += outcome.get("plan_bonus", 0.0)
        r += outcome.get("memory_bonus", 0.0)

        potential_delta = float(outcome.get("milestone_potential_delta", 0.0))
        if potential_delta >= 0.0:
            r += min(0.05, potential_delta * 0.45)
        else:
            r += max(-0.04, potential_delta * 0.30)

        token_cost = int(outcome.get("token_cost", 0) or 0)
        efficiency_before = self._token_efficiency_score
        token_penalty = min(0.05, token_cost / max(1, self._token_budget_total) * 1.2)
        r -= token_penalty
        if outcome.get("enrolled") or outcome.get("screen_success") or outcome.get("memory_hit"):
            progress_bonus = min(0.04, self._token_efficiency_score * 0.03)
            r += progress_bonus
            outcome["token_bonus"] += round(progress_bonus - token_penalty, 4)
        else:
            outcome["token_bonus"] -= round(token_penalty, 4)

        # --- UPGRADE ADDITIONS ---

        # Hypothesis bonus (Upgrade 1)
        r += self._hypothesis_bonus(hypothesis)

        # Step pressure (Upgrade 4): increasing urgency over time
        step_penalty = 0.005 * (self._step / self._max_steps)
        r -= step_penalty

        # Consistency penalty (Upgrade 2)
        r -= self._consistency_penalty()

        # Confidence calibration on finalize (Upgrade 4)
        if action_type == "stop_recruitment":
            # Compute an approval score based on enrollment progress
            enrollment_score = self._enrolled / max(1, self._target)
            confidence_penalty = abs(confidence - enrollment_score)
            r -= confidence_penalty * 0.3

            # Uncertainty penalty: penalize stopping when things are still uncertain
            if self._uncertainty > 0.3:
                r -= 0.15

        # Planning/memory actions already penalized in FIX 4 above (-0.06 to -0.07).
        # Removed duplicate -0.02 opportunity cost here that caused double-penalty
        # totaling -0.08 to -0.09 per planning step, which was too harsh.

        self._token_efficiency_score = self._compute_token_efficiency()
        if self._token_efficiency_score < efficiency_before and not outcome.get("enrolled"):
            r -= min(0.02, (efficiency_before - self._token_efficiency_score) * 0.1)

        return max(-0.5, min(0.99, r))

    def _make_observation(self) -> Observation:
        # Available patients: up to 5 unscreened candidates
        available = []
        candidates = [
            p for p in self._patients if not p["screened"] and not p["dropped"]
        ]
        candidates.sort(key=lambda p: p.get("priority_score", 0.0), reverse=True)
        for p in candidates[:5]:
            available.append(
                {
                    "id": p["id"],
                    "age": p["age"],
                    "eligibility_score": p["eligibility_score"],
                    "dropout_risk": p["dropout_risk"],
                }
            )

        recontact_candidates = []
        recontact_pool = [
            p
            for p in self._patients
            if p["screened"] and not p["enrolled"] and not p["dropped"]
        ]
        recontact_pool.sort(
            key=lambda p: (
                int(p.get("followup_due_day") is not None),
                -(p.get("followup_due_day") or 10**6),
                p.get("priority_score", 0.0),
            ),
            reverse=True,
        )
        for p in recontact_pool[:5]:
            recontact_candidates.append(
                {
                    "id": p["id"],
                    "age": p["age"],
                    "eligible": bool(p.get("eligible", False)),
                    "consented": bool(p.get("consented", False)),
                    "dropout_risk": p["dropout_risk"],
                    "followup_due_day": p.get("followup_due_day"),
                }
            )

        allocation_candidates = []
        allocation_pool = [
            p
            for p in self._patients
            if p.get("consented") and not p.get("enrolled") and not p.get("dropped")
        ]
        allocation_pool.sort(
            key=lambda p: (
                p.get("priority_score", 0.0),
                -p.get("dropout_risk", 0.0),
            ),
            reverse=True,
        )
        for p in allocation_pool[:5]:
            allocation_candidates.append(
                {
                    "id": p["id"],
                    "age": p["age"],
                    "dropout_risk": p["dropout_risk"],
                    "followup_due_day": p.get("followup_due_day"),
                }
            )

        # Site performance summary
        site_perf = {}
        for sid, site in self._sites.items():
            site_perf[sid] = {
                "conversion_rate": site["conversion_rate"],
                "avg_wait_days": site["avg_wait_days"],
                "capacity_remaining": site["capacity_remaining"],
                "retention_rate": site.get("retention_rate", 0.0),
            }

        # Recent events
        recent = []
        for h in self._history[-5:]:
            if h.get("dropout"):
                recent.append("patient_dropout")
            if h.get("delayed_effects_triggered"):
                recent.append("delayed_effect")
            if h.get("milestone_bonus", 0.0) > 0:
                recent.append("enrollment_milestone")
            if h.get("error"):
                recent.append(f"error:{h['error']}")
        # Use previous step's events so we don't leak future events into the observation
        day_events = self._events.get(max(0, self._step - 1), [])
        recent.extend(day_events[:3])

        # 7-day rolling dropout rate
        dropout_7d = self._rolling_dropout_rate()

        # --- UPGRADE 3: Causal feedback layer ---
        dominant = "unclear"
        if dropout_7d > 0.3:
            dominant = "dropout-related effects"
        elif self._uncertainty > 0.3:
            dominant = "noise-related variability"
        else:
            # Check site variance
            conv_rates = [s.get("conversion_rate", 0.5) for s in site_perf.values()]
            if conv_rates and len(conv_rates) > 1:
                site_variance = max(conv_rates) - min(conv_rates)
                if site_variance > 0.2:
                    dominant = "site performance bias"
                else:
                    dominant = "possible confounding factors"
            else:
                dominant = "possible confounding factors"

        causal_insight = (
            f"Insight: Observed changes may be driven by {dominant}. "
            f"Dropout rate={dropout_7d:.2f}, uncertainty={self._uncertainty:.2f}. "
            f"However, confounding may distort this signal."
        )

        # --- Hypothesis accuracy feedback ---
        hyp_accuracy = 0.0
        hypothesis_mapping = {
            "noise_dominant": "noise",
            "dropout_dominant": "dropout",
            "site_bias": "site_bias",
        }
        if hypothesis_mapping.get(self._last_hypothesis, "") == self._world_type:
            hyp_accuracy = 1.0
        elif self._last_hypothesis != "unknown":
            # Partial credit for related hypotheses
            related = {
                ("noise_dominant", "noise"): 0.8,
                ("dropout_dominant", "dropout"): 0.8,
                ("site_bias", "site_bias"): 1.0,
                ("confounding", "dropout"): 0.3,
                ("confounding", "noise"): 0.3,
            }
            hyp_accuracy = related.get((self._last_hypothesis, self._world_type), 0.1)

        return Observation(
            timestamp=self._step,
            budget_remaining=round(self._budget_remaining, 2),
            time_to_deadline_days=max(0, self._deadline_days - self._step),
            enrolled_so_far=self._enrolled,
            target_enrollment=self._target,
            task_id=self._task or "",
            max_steps=self._max_steps,
            initial_budget=round(float(self._trace.get("budget", 0.0)), 2),
            current_funnel=dict(self._funnel),
            available_patients=available,
            recontact_candidates=recontact_candidates,
            allocation_candidates=allocation_candidates,
            site_performance=site_perf,
            recent_events=recent[-5:],
            uncertainty_level=round(self._uncertainty, 3),
            difficulty=self._difficulty(),
            dropout_rate_7d=round(dropout_7d, 3),
            screening_backlog=self._screening_backlog,
            causal_insight=causal_insight,
            hypothesis_accuracy=round(hyp_accuracy, 3),
            world_type=self._world_type,
            milestones=dict(self._milestones),
            active_constraints=dict(self._active_constraints),
            delayed_effects_pending=len(self._delayed_effects),
            uncertainty_components=dict(self._uncertainty_components),
            patient_memory_summary=dict(self._patient_memory),
            counterfactual_hint=self._counterfactual_hint,
            current_plan=self._current_plan_snapshot(),
            indexed_memory_summary=self._summarize_indexed_memory(),
            retrieved_memory_context=self._retrieved_memory_context,
            milestone_potential=round(self._milestone_potential, 3),
            active_milestone=self._active_milestone,
            hindsight_available=bool(self._hindsight_summary),
            token_budget_remaining=self._token_budget_remaining,
            token_usage_so_far=self._token_usage_so_far,
            token_efficiency_score=round(self._token_efficiency_score, 3),
            counterfactual_rollout=self._counterfactual_rollout_summary(),
        )

    def _difficulty(self) -> int:
        task = self._task or ""
        if task.startswith("hard_bench"):
            return 3
        elif task.startswith("medium_bench"):
            return 2
        return 1
