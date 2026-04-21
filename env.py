"""Core Adaptive Clinical Trial Recruitment environment implementing step/reset/state.

Simulates a 180-day clinical trial recruitment period where agents optimize
the screening -> enrollment -> retention funnel under budget, time pressure,
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
  - Step pressure: -0.03 * step (increasing urgency)
  - Confidence calibration: penalizes overconfident/underconfident finalize
"""

import copy
import random
from typing import Optional, Dict, Any, List

from models import Observation, Action, State, StepResult
from load_traces import TASK_TRACES
from graders import GRADERS


class ClinicalRecruitmentEnv:
    """Clinical trial recruitment optimization environment.

    The agent manages the patient recruitment funnel:
    screening, site allocation, strategy adjustment, and retention
    across a 180-day trial recruitment window.
    """

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

    def reset(self, task: Optional[str] = None) -> StepResult:
        if task is None:
            task = "easy_bench"
        if task not in TASK_TRACES:
            raise ValueError(
                f"Unknown task: {task}. Available: {list(TASK_TRACES.keys())}"
            )

        self._task = task
        self._trace = TASK_TRACES[task]
        self._step = 0
        self._max_steps = 180
        self._done = False
        self._total_reward = 0.0
        self._history = []

        # Seed RNG deterministically per task
        seeds = {"easy_bench": 42, "medium_bench": 123, "hard_bench": 777}
        self._rng = random.Random(seeds.get(task, 42))

        # Deep-copy mutable state from trace
        self._patients = copy.deepcopy(self._trace["patients"])
        self._sites = copy.deepcopy(self._trace["sites"])
        self._events = copy.deepcopy(self._trace["events"])
        self._curriculum = copy.deepcopy(self._trace.get("curriculum", []))

        # Trial parameters
        self._budget_remaining = self._trace["budget"]
        self._target = self._trace["target_enrollment"]
        self._deadline_days = self._trace["deadline_days"]
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
        }

        action_error = None

        # -- Track hypothesis from agent (Upgrade 1) --
        hypothesis = action.hypothesis or "unknown"
        confidence = action.confidence
        self._hypothesis_history.append(hypothesis)
        self._last_hypothesis = hypothesis
        self._last_confidence = confidence

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
                "enrolled": self._enrolled,
                "budget": round(self._budget_remaining, 2),
                "reward": round(reward, 4),
                "dropout": outcome["dropout"],
                "screen_success": outcome["screen_success"],
                "curriculum": in_curriculum,
                "milestone_bonus": round(outcome["milestone_bonus"], 3),
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
                "delayed_effects_triggered": outcome["delayed_effects_triggered"],
            },
            "last_action_error": action_error,
        }

        # If done, run grader
        if self._done:
            grader = GRADERS.get(self._task)
            if grader:
                final_score = grader(obs, self._total_reward, self._history)
                # Strict clamp: must be in (0, 1), never exactly 0.0 or 1.0
                final_score = min(0.999, max(0.001, float(final_score)))
                info["final_score"] = final_score
                info["grader_task"] = self._task

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
            done=self._done,
            enrolled_so_far=self._enrolled,
            target_enrollment=self._target,
            budget_remaining=round(self._budget_remaining, 2),
            total_reward=round(self._total_reward, 4),
            history=self._history[-10:],
            milestones=dict(self._milestones),
            active_constraints=dict(self._active_constraints),
            delayed_effects_pending=len(self._delayed_effects),
            uncertainty_components=dict(self._uncertainty_components),
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
                self._schedule_delayed_effect(
                    2,
                    {"type": "site_negotiation", "site_id": site_key},
                )
            else:
                error = "site_not_found"
        else:
            error = "unknown_strategy"

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
        elif effect_type == "site_negotiation":
            site_id = effect.get("site_id")
            site = self._sites.get(site_id or "")
            if site:
                site["conversion_rate"] = round(
                    min(0.98, site["conversion_rate"] + 0.04),
                    3,
                )
                site["avg_wait_days"] = round(max(1.0, site["avg_wait_days"] - 1.0), 1)
                site["capacity_remaining"] = min(
                    site["capacity_total"],
                    site["capacity_remaining"] + 2,
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
        self._counterfactual_hint = self._build_counterfactual_hint()

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
        return "Counterfactual: keep screening high-priority patients while budget remains."

    def _recompute_patient_priority(self, patient: Dict[str, Any]):
        patient["priority_score"] = round(
            patient["eligibility_score"]
            * patient["willingness"]
            * (1.0 - patient["dropout_risk"]),
            3,
        )

    def _consistency_penalty(self) -> float:
        """Upgrade 2: Penalize erratic hypothesis switching.
        If the agent switches hypothesis more than 2 times, apply penalty.
        """
        if len(self._hypothesis_history) < 2:
            return 0.0
        switches = 0
        for i in range(1, len(self._hypothesis_history)):
            if self._hypothesis_history[i] != self._hypothesis_history[i - 1]:
                switches += 1
        if switches > 2:
            return 0.10
        return 0.0

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
        if action_type in ("screen_patient", "recontact", "allocate_to_site"):
            # Productive actions: no penalty
            pass
        elif action_type == "adjust_strategy":
            # Strategy is fine but not productive by itself
            r -= 0.02
        elif action_type == "stop_recruitment":
            # Handled below in confidence calibration
            pass
        else:
            # Unknown action
            r -= 0.05

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
        day_events = self._events.get(self._step, [])
        recent.extend(day_events[:3])

        # 7-day rolling dropout rate
        recent_drops = [d for d in self._recent_dropouts if d >= self._step - 7]
        enrolled_count = max(1, self._enrolled)
        dropout_7d = (
            min(1.0, len(recent_drops) / enrolled_count) if enrolled_count > 0 else 0.0
        )

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
            current_funnel=dict(self._funnel),
            available_patients=available,
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
        )

    def _difficulty(self) -> int:
        if self._task == "hard_bench":
            return 3
        elif self._task == "medium_bench":
            return 2
        return 1
