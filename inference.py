"""Baseline inference script for Adaptive Clinical Trial Recruitment environment."""

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# -- Environment variables (checklist-compliant) --
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv(
    "ENV_URL", "https://pratimassaravanan-clinical-recruitment.hf.space"
)
BENCHMARK = "adaptive-clinical-recruitment"
LLM_CALL_INTERVAL = 5
TEMPERATURE = 0.0
MAX_TOTAL_REWARD = 180.0
SUCCESS_SCORE_THRESHOLD = 0.5


# -- Structured logging (matches sample format exactly) --
def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(error: Optional[str]) -> str:
    if error is None:
        return "null"
    cleaned = str(error).replace("\r", " ").replace("\n", " ").strip()
    return cleaned or "null"


def _format_action(action: dict) -> str:
    atype = action.get("action_type", "screen_patient")
    pid = action.get("patient_id", "")
    sid = action.get("site_id", "")
    strat = action.get("strategy_change", "")
    target_phase = action.get("target_phase", "")
    memory_key = action.get("memory_key", "")
    memory_query = action.get("memory_query", "")
    token_cost = action.get("token_cost", "")
    hyp = action.get("hypothesis", "")
    parts = [atype]
    if pid:
        parts.append(pid)
    if sid:
        parts.append(sid)
    if strat:
        parts.append(strat)
    if target_phase:
        parts.append(f"phase={target_phase}")
    if memory_key:
        parts.append(f"mem={memory_key}")
    if memory_query:
        parts.append(f"query={memory_query}")
    if token_cost not in (None, ""):
        parts.append(f"tok={token_cost}")
    if hyp:
        parts.append(f"hyp={hyp}")
    return "/".join(parts)


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_format_bool(done)} error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={_format_bool(success)} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# -- Environment client --
class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60)

    def reset(self, task_id: str) -> dict:
        r = self.http.post(f"{self.base}/reset", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.http.post(f"{self.base}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.http.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.http.close()


class PolicyState:
    """Tracks actionable patient memory across steps for the baseline policy."""

    def __init__(self):
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.last_strategy_step: Dict[str, int] = {}
        self.last_plan_step: int = -999
        self.last_memory_write_step: int = -999
        self.last_memory_retrieval_step: int = -999
        self.token_usage_estimate: int = 0

    def reset(self, obs: dict) -> None:
        self.patients = {}
        self.last_strategy_step = {}
        self.last_plan_step = -999
        self.last_memory_write_step = -999
        self.last_memory_retrieval_step = -999
        self.token_usage_estimate = 0
        self._remember_available_patients(obs)

    def _ensure_patient(self, patient_id: str) -> Dict[str, Any]:
        return self.patients.setdefault(
            patient_id,
            {
                "status": "unknown",
                "eligible": False,
                "screened": False,
                "followup_due_step": None,
                "priority": 0.0,
                "dropout_risk": 0.0,
                "eligibility_score": 0.0,
                "site_id": None,
                "recontact_attempts": 0,
            },
        )

    def _priority(self, patient: Dict[str, Any]) -> float:
        return round(
            float(patient.get("eligibility_score", 0.0))
            * (1.0 - float(patient.get("dropout_risk", 0.0)) * 0.8),
            4,
        )

    def _remember_available_patients(self, obs: dict) -> None:
        for patient in obs.get("available_patients", []):
            pid = patient.get("id")
            if not pid:
                continue
            entry = self._ensure_patient(pid)
            entry["eligibility_score"] = float(patient.get("eligibility_score", 0.0))
            entry["dropout_risk"] = float(patient.get("dropout_risk", 0.0))
            entry["priority"] = self._priority(patient)
            if entry.get("status") in ("unknown", "available"):
                entry["status"] = "available"

    def strategy_recently_used(
        self, strategy_change: str, step_num: int, cooldown: int = 10
    ) -> bool:
        return step_num - self.last_strategy_step.get(strategy_change, -999) < cooldown

    def record_strategy(self, strategy_change: Optional[str], step_num: int) -> None:
        if strategy_change:
            self.last_strategy_step[strategy_change] = step_num

    def planning_recently_used(self, step_num: int, cooldown: int = 8) -> bool:
        return step_num - self.last_plan_step < cooldown

    def memory_write_recently_used(self, step_num: int, cooldown: int = 10) -> bool:
        return step_num - self.last_memory_write_step < cooldown

    def memory_retrieval_recently_used(self, step_num: int, cooldown: int = 6) -> bool:
        return step_num - self.last_memory_retrieval_step < cooldown

    def tracked_patients(self, step_num: int, status: str) -> List[str]:
        candidates = []
        for patient_id, entry in self.patients.items():
            if entry.get("status") != status:
                continue
            due = entry.get("followup_due_step")
            due_rank = due if isinstance(due, int) else 10**6
            candidates.append((due_rank, -float(entry.get("priority", 0.0)), patient_id))
        candidates.sort()
        return [patient_id for _, _, patient_id in candidates]

    def consented_pending_ids(self, step_num: int) -> List[str]:
        return self.tracked_patients(step_num, "consented_pending")

    def recontact_candidate_ids(self, step_num: int) -> List[str]:
        actionable = []
        for patient_id, entry in self.patients.items():
            if entry.get("status") != "eligible_pending":
                continue
            due = entry.get("followup_due_step")
            due_rank = due if isinstance(due, int) else 10**6
            actionable.append((due_rank, entry.get("recontact_attempts", 0), -float(entry.get("priority", 0.0)), patient_id))
        actionable.sort()
        return [patient_id for _, _, _, patient_id in actionable]

    def describe_tracked_patients(self, step_num: int, limit: int = 8) -> str:
        lines = []
        ranked = []
        for patient_id, entry in self.patients.items():
            status = entry.get("status")
            if status in {"unknown", "available", "screen_failed"}:
                continue
            due = entry.get("followup_due_step")
            due_rank = due if isinstance(due, int) else 10**6
            ranked.append((due_rank, -float(entry.get("priority", 0.0)), patient_id, entry))
        ranked.sort()
        for _, _, patient_id, entry in ranked[:limit]:
            due = entry.get("followup_due_step")
            due_text = due if isinstance(due, int) else "-"
            lines.append(
                f"  {patient_id}: status={entry.get('status')}, priority={entry.get('priority', 0):.2f}, due={due_text}, dropout={entry.get('dropout_risk', 0):.2f}"
            )
        return "\n".join(lines) if lines else "  (no tracked actionable patients yet)"

    def update(self, prev_obs: dict, action: dict, result: dict, step_num: int) -> None:
        next_obs = result.get("observation", {})
        info = result.get("info", {})
        reward_breakdown = info.get("reward_breakdown", {})
        error = info.get("last_action_error")
        action_type = action.get("action_type")
        patient_id = action.get("patient_id")
        self.record_strategy(action.get("strategy_change"), step_num)
        if action_type == "plan_next_phase":
            self.last_plan_step = step_num
        elif action_type == "summarize_and_index":
            self.last_memory_write_step = step_num
        elif action_type == "retrieve_relevant_history":
            self.last_memory_retrieval_step = step_num
        self.token_usage_estimate += int(action.get("token_cost", 0) or 0)

        self._remember_available_patients(prev_obs)
        self._remember_available_patients(next_obs)

        prev_funnel = prev_obs.get("current_funnel", {})
        next_funnel = next_obs.get("current_funnel", {})
        consent_delta = int(next_funnel.get("consented", 0)) - int(
            prev_funnel.get("consented", 0)
        )
        enrolled_delta = int(next_funnel.get("enrolled", 0)) - int(
            prev_funnel.get("enrolled", 0)
        )
        dropped_delta = int(next_funnel.get("dropped", 0)) - int(
            prev_funnel.get("dropped", 0)
        )

        if patient_id:
            entry = self._ensure_patient(patient_id)
            if error == "already_enrolled":
                entry["status"] = "enrolled"
            elif error == "patient_not_consented":
                entry["status"] = "eligible_pending"
                entry["followup_due_step"] = step_num + 2
            elif error == "patient_dropped":
                entry["status"] = "dropped"
            elif error == "already_screened":
                entry["screened"] = True
            elif error is None and action_type == "screen_patient":
                entry["screened"] = True
                if reward_breakdown.get("screen_success"):
                    entry["eligible"] = True
                    if consent_delta > 0:
                        entry["status"] = "consented_pending"
                        entry["followup_due_step"] = step_num + 5
                    else:
                        entry["status"] = "eligible_pending"
                        entry["followup_due_step"] = step_num + 3
                else:
                    entry["status"] = "screen_failed"
                    entry["followup_due_step"] = None
            elif error is None and action_type == "recontact":
                entry["recontact_attempts"] = int(entry.get("recontact_attempts", 0)) + 1
                if enrolled_delta > 0:
                    entry["status"] = "enrolled"
                    entry["followup_due_step"] = None
                elif consent_delta > 0:
                    entry["status"] = "consented_pending"
                    entry["followup_due_step"] = step_num + 4
                elif entry.get("eligible"):
                    entry["status"] = "eligible_pending"
                    entry["followup_due_step"] = step_num + 2
            elif error is None and action_type == "allocate_to_site":
                entry["site_id"] = action.get("site_id")
                if enrolled_delta > 0:
                    entry["status"] = "enrolled"
                    entry["followup_due_step"] = None
                elif dropped_delta > 0:
                    entry["status"] = "dropped"
                    entry["followup_due_step"] = None

        if consent_delta < 0:
            to_expire = abs(consent_delta)
            for tracked_id in self.consented_pending_ids(step_num):
                entry = self.patients.get(tracked_id, {})
                due = entry.get("followup_due_step")
                if isinstance(due, int) and due <= step_num:
                    entry["status"] = "eligible_pending"
                    entry["followup_due_step"] = step_num + 2
                    to_expire -= 1
                    if to_expire <= 0:
                        break

        if enrolled_delta < 0:
            to_drop = abs(enrolled_delta)
            ranked = []
            for tracked_id, entry in self.patients.items():
                if entry.get("status") == "enrolled":
                    ranked.append((-float(entry.get("dropout_risk", 0.0)), tracked_id))
            ranked.sort()
            for _, tracked_id in ranked[:to_drop]:
                self.patients[tracked_id]["status"] = "dropped"
                self.patients[tracked_id]["followup_due_step"] = None


def _site_suffix(site_id: Optional[str]) -> str:
    if not site_id:
        return ""
    return str(site_id).replace("site_", "")


def _valid_strategy_choices(obs: dict) -> List[str]:
    choices = ["increase_outreach", "relax_criteria", "tighten_criteria"]
    for site_id in obs.get("site_performance", {}):
        suffix = _site_suffix(site_id)
        if suffix:
            choices.append(f"focus_site_{suffix}")
            choices.append(f"negotiate_site_{suffix}")
    return choices


def _best_site(obs: dict, mode: str = "allocate") -> Optional[str]:
    sites = obs.get("site_performance", {})
    best_site_id = None
    best_score = -1.0
    for site_id, site in sites.items():
        capacity = float(site.get("capacity_remaining", 0))
        if capacity <= 0:
            continue
        conversion = float(site.get("conversion_rate", 0.0))
        wait_days = float(site.get("avg_wait_days", 0.0))
        retention = float(site.get("retention_rate", 0.8))
        if mode == "negotiate":
            score = conversion * 1.5 + wait_days * 0.2 + max(0.0, 6.0 - capacity) * 0.1
        else:
            score = conversion * retention * (0.5 + capacity / 10.0) / max(1.0, wait_days)
        if score > best_score:
            best_score = score
            best_site_id = site_id
    return best_site_id


def _best_available_patient(obs: dict) -> Optional[str]:
    available = obs.get("available_patients", [])
    if not available:
        return None
    patient = max(
        available,
        key=lambda item: float(item.get("eligibility_score", 0.0))
        * (1.0 - float(item.get("dropout_risk", 0.0)) * 0.8),
    )
    return patient.get("id")


def _normalize_action(
    action: dict, obs: dict, step_num: int, policy_state: PolicyState
) -> Optional[dict]:
    valid_actions = {
        "screen_patient",
        "recontact",
        "allocate_to_site",
        "adjust_strategy",
        "plan_next_phase",
        "summarize_and_index",
        "retrieve_relevant_history",
        "stop_recruitment",
    }
    action_type = action.get("action_type")
    if action_type not in valid_actions:
        return None

    normalized = {
        "action_type": action_type,
        "patient_id": action.get("patient_id"),
        "site_id": action.get("site_id"),
        "strategy_change": action.get("strategy_change"),
        "hypothesis": action.get("hypothesis"),
        "confidence": action.get("confidence", 0.5),
        "plan_id": action.get("plan_id"),
        "plan_summary": action.get("plan_summary"),
        "target_phase": action.get("target_phase"),
        "memory_key": action.get("memory_key"),
        "memory_query": action.get("memory_query"),
        "memory_payload": action.get("memory_payload"),
        "token_cost": action.get("token_cost"),
    }

    for key in (
        "patient_id",
        "site_id",
        "strategy_change",
        "plan_id",
        "plan_summary",
        "target_phase",
        "memory_key",
        "memory_query",
        "memory_payload",
    ):
        if normalized.get(key) in (None, "null", "None", ""):
            normalized[key] = None

    token_cost = normalized.get("token_cost")
    if isinstance(token_cost, (int, float)):
        normalized["token_cost"] = max(0, int(token_cost))
    else:
        normalized["token_cost"] = None

    valid_hypotheses = {
        "dropout_dominant",
        "noise_dominant",
        "site_bias",
        "confounding",
        "unknown",
    }
    if normalized.get("hypothesis") not in valid_hypotheses:
        normalized["hypothesis"] = _infer_hypothesis(obs)

    confidence = normalized.get("confidence", 0.5)
    if isinstance(confidence, (int, float)):
        normalized["confidence"] = max(0.0, min(1.0, float(confidence)))
    else:
        normalized["confidence"] = _infer_confidence(obs, step_num)

    if action_type == "screen_patient":
        normalized["patient_id"] = normalized.get("patient_id") or _best_available_patient(obs)
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        if not normalized["patient_id"]:
            return None
    elif action_type == "recontact":
        candidates = policy_state.recontact_candidate_ids(step_num)
        normalized["patient_id"] = normalized.get("patient_id") or (
            candidates[0] if candidates else None
        )
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        if not normalized["patient_id"]:
            return None
    elif action_type == "allocate_to_site":
        candidates = policy_state.consented_pending_ids(step_num)
        normalized["patient_id"] = normalized.get("patient_id") or (
            candidates[0] if candidates else None
        )
        normalized["site_id"] = normalized.get("site_id") or _best_site(
            obs, mode="allocate"
        )
        normalized["strategy_change"] = None
        if not normalized["patient_id"] or not normalized["site_id"]:
            return None
    elif action_type == "adjust_strategy":
        valid_strategies = set(_valid_strategy_choices(obs))
        if normalized.get("strategy_change") not in valid_strategies:
            return None
        normalized["patient_id"] = None
        normalized["site_id"] = None
        normalized["plan_id"] = None
        normalized["plan_summary"] = None
        normalized["target_phase"] = None
        normalized["memory_key"] = None
        normalized["memory_query"] = None
        normalized["memory_payload"] = None
    elif action_type == "plan_next_phase":
        valid_phases = {"screening", "conversion", "allocation", "retention", "recovery"}
        phase = normalized.get("target_phase") or _recommended_phase(obs)
        if phase not in valid_phases:
            return None
        normalized["target_phase"] = phase
        normalized["plan_id"] = normalized.get("plan_id") or f"plan-{step_num}-{phase}"
        normalized["plan_summary"] = normalized.get("plan_summary") or _plan_summary_for_phase(
            phase,
            obs,
        )
        normalized["patient_id"] = None
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        normalized["memory_key"] = None
        normalized["memory_query"] = None
        normalized["memory_payload"] = None
    elif action_type == "summarize_and_index":
        key = normalized.get("memory_key") or _default_memory_key(obs)
        payload = normalized.get("memory_payload") or _default_memory_payload(obs)
        if not key or not payload:
            return None
        normalized["memory_key"] = key
        normalized["memory_payload"] = payload
        normalized["patient_id"] = None
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        normalized["plan_id"] = None
        normalized["plan_summary"] = None
        normalized["target_phase"] = None
        normalized["memory_query"] = None
    elif action_type == "retrieve_relevant_history":
        query = normalized.get("memory_query") or _default_memory_query(obs)
        if not query:
            return None
        normalized["memory_query"] = query
        normalized["patient_id"] = None
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        normalized["plan_id"] = None
        normalized["plan_summary"] = None
        normalized["target_phase"] = None
        normalized["memory_key"] = None
        normalized["memory_payload"] = None
    else:
        normalized["patient_id"] = None
        normalized["site_id"] = None
        normalized["strategy_change"] = None
        normalized["plan_id"] = None
        normalized["plan_summary"] = None
        normalized["target_phase"] = None
        normalized["memory_key"] = None
        normalized["memory_query"] = None
        normalized["memory_payload"] = None
        progress = obs.get("enrolled_so_far", 0) / max(1, obs.get("target_enrollment", 1))
        normalized["confidence"] = min(normalized["confidence"], max(0.1, progress))

    return normalized


def _estimated_token_cost(action_type: str) -> int:
    return {
        "screen_patient": 18,
        "recontact": 20,
        "allocate_to_site": 22,
        "adjust_strategy": 26,
        "plan_next_phase": 55,
        "summarize_and_index": 42,
        "retrieve_relevant_history": 36,
        "stop_recruitment": 12,
    }.get(action_type, 20)


# -- Rule-based fallback policy --
def _infer_hypothesis(obs: dict) -> str:
    """Infer a hypothesis about dominant trial dynamics from observation."""
    dropout_7d = obs.get("dropout_rate_7d", 0)
    uncertainty = obs.get("uncertainty_level", 0)
    sites = obs.get("site_performance", {})
    uncertainty_components = obs.get("uncertainty_components", {})
    patient_memory = obs.get("patient_memory_summary", {})
    constraints = obs.get("active_constraints", {})

    patient_uncertainty = float(uncertainty_components.get("patient_pool", uncertainty))
    site_uncertainty = float(uncertainty_components.get("site_operations", 0.0))
    policy_uncertainty = float(uncertainty_components.get("policy", 0.0))
    at_risk_enrolled = int(patient_memory.get("at_risk_enrolled", 0))

    if dropout_7d > 0.25 or (dropout_7d > 0.12 and at_risk_enrolled >= 3):
        return "dropout_dominant"

    # Check site variance
    conv_rates = [s.get("conversion_rate", 0.5) for s in sites.values()]
    if len(conv_rates) > 1:
        site_var = max(conv_rates) - min(conv_rates)
        if (
            site_var > 0.18
            or site_uncertainty > 0.45
            or constraints.get("site_bottleneck", False)
        ):
            return "site_bias"

    if patient_uncertainty > max(site_uncertainty, policy_uncertainty, 0.35):
        return "noise_dominant"

    if uncertainty > 0.4 and policy_uncertainty < 0.35:
        return "noise_dominant"

    return "confounding"


def _infer_confidence(obs: dict, step: int) -> float:
    """Estimate confidence based on how much data we've seen."""
    # Confidence grows with steps (more data = more certainty)
    base = min(0.92, 0.38 + step * 0.006 + obs.get("hypothesis_accuracy", 0) * 0.12)
    # Lower confidence if uncertainty is high
    uncertainty_components = obs.get("uncertainty_components", {})
    component_values = [
        float(value)
        for value in uncertainty_components.values()
        if isinstance(value, (int, float))
    ]
    if component_values:
        base -= (sum(component_values) / len(component_values)) * 0.15
    base -= obs.get("uncertainty_level", 0) * 0.15
    base -= min(0.12, obs.get("delayed_effects_pending", 0) * 0.02)
    if obs.get("active_constraints", {}).get("regulatory_hold_days", 0) > 0:
        base -= 0.05
    return max(0.1, min(0.95, base))


def _recommended_phase(obs: dict) -> str:
    constraints = obs.get("active_constraints", {})
    memory = obs.get("patient_memory_summary", {})
    if int(constraints.get("regulatory_hold_days", 0)) > 0:
        return "recovery"
    if int(memory.get("consented_pending_allocation", 0)) > 0:
        return "allocation"
    if int(memory.get("followup_due", 0)) > 0 or int(
        memory.get("eligible_pending_consent", 0)
    ) > 0:
        return "conversion"
    if int(memory.get("at_risk_enrolled", 0)) > 0 and obs.get("difficulty") == 3:
        return "retention"
    return "screening"


def _plan_summary_for_phase(phase: str, obs: dict) -> str:
    active_milestone = obs.get("active_milestone") or "next"
    summaries = {
        "screening": f"Push high-priority screening to reach the {active_milestone} frontier.",
        "conversion": f"Convert warm leads before follow-up windows close and unlock the {active_milestone} milestone.",
        "allocation": f"Allocate consented patients quickly to preserve momentum toward the {active_milestone} milestone.",
        "retention": f"Protect fragile enrollments so progress to {active_milestone} does not slip backward.",
        "recovery": f"Stabilize constraints and recover pipeline health before chasing the {active_milestone} frontier.",
    }
    return summaries.get(phase, "Advance the next long-horizon phase.")


def _default_memory_key(obs: dict) -> str:
    phase = _recommended_phase(obs)
    milestone = obs.get("active_milestone") or "frontier"
    return f"{phase}_{milestone}"


def _default_memory_payload(obs: dict) -> str:
    memory = obs.get("patient_memory_summary", {})
    constraints = obs.get("active_constraints", {})
    return (
        f"phase={_recommended_phase(obs)} active_milestone={obs.get('active_milestone', '')} "
        f"followup_due={memory.get('followup_due', 0)} consented_pending={memory.get('consented_pending_allocation', 0)} "
        f"reg_hold={constraints.get('regulatory_hold_days', 0)} site_bottleneck={constraints.get('site_bottleneck', False)}"
    )


def _default_memory_query(obs: dict) -> str:
    phase = _recommended_phase(obs)
    if phase == "allocation":
        return "allocation consented site"
    if phase == "conversion":
        return "conversion followup consent"
    if phase == "retention":
        return "retention dropout recovery"
    if phase == "recovery":
        return "recovery regulatory site bottleneck"
    return "screening high priority"


def _hard_mode_action(
    obs: dict,
    step: int,
    policy_state: PolicyState,
    hypothesis: str,
    confidence: float,
) -> Optional[dict]:
    """Retention-heavy serving policy chosen from offline research results.

    Offline runs showed the conservative retention policy outperformed the generic
    memory baseline on `hard_bench`, so hard-mode routing prefers follow-up,
    stability, and selective screening over aggressive outreach churn.
    """

    budget = obs.get("budget_remaining", 0)
    time_left = obs.get("time_to_deadline_days", 180)
    constraints = obs.get("active_constraints", {})
    available = obs.get("available_patients", [])
    uncertainty_components = obs.get("uncertainty_components", {})
    patient_uncertainty = float(
        uncertainty_components.get("patient_pool", obs.get("uncertainty_level", 0.0))
    )
    regulatory_hold_days = int(constraints.get("regulatory_hold_days", 0))
    delayed_effects_pending = int(obs.get("delayed_effects_pending", 0))
    milestone_potential = float(obs.get("milestone_potential", 0.0) or 0.0)
    indexed_memory_summary = obs.get("indexed_memory_summary", {})
    retrieved_memory_context = obs.get("retrieved_memory_context", "")
    current_plan = obs.get("current_plan", {})
    recontact_ids = policy_state.recontact_candidate_ids(step)
    consented_ids = policy_state.consented_pending_ids(step)
    best_site = _best_site(obs, mode="allocate")
    best_negotiate_site = _best_site(obs, mode="negotiate")
    recommended_phase = _recommended_phase(obs)

    def emit(
        action_type: str,
        patient_id: Optional[str] = None,
        site_id: Optional[str] = None,
        strategy_change: Optional[str] = None,
        conf: Optional[float] = None,
        plan_id: Optional[str] = None,
        plan_summary: Optional[str] = None,
        target_phase: Optional[str] = None,
        memory_key: Optional[str] = None,
        memory_query: Optional[str] = None,
        memory_payload: Optional[str] = None,
    ) -> dict:
        return {
            "action_type": action_type,
            "patient_id": patient_id,
            "site_id": site_id,
            "strategy_change": strategy_change,
            "hypothesis": hypothesis,
            "confidence": confidence if conf is None else conf,
            "plan_id": plan_id,
            "plan_summary": plan_summary,
            "target_phase": target_phase,
            "memory_key": memory_key,
            "memory_query": memory_query,
            "memory_payload": memory_payload,
            "token_cost": _estimated_token_cost(action_type),
        }

    if (
        current_plan.get("target_phase") != recommended_phase
        and not policy_state.planning_recently_used(step, cooldown=10)
    ):
        return emit(
            "plan_next_phase",
            target_phase=recommended_phase,
            plan_id=f"hard-plan-{step}-{recommended_phase}",
            plan_summary=_plan_summary_for_phase(recommended_phase, obs),
            conf=max(confidence, 0.62),
        )

    if (
        indexed_memory_summary.get("entries", 0) <= 0
        and not policy_state.memory_write_recently_used(step, cooldown=12)
    ):
        return emit(
            "summarize_and_index",
            memory_key=_default_memory_key(obs),
            memory_payload=_default_memory_payload(obs),
            conf=max(0.45, confidence - 0.08),
        )

    if (
        indexed_memory_summary.get("entries", 0) > 0
        and not retrieved_memory_context
        and milestone_potential < 0.65
        and not policy_state.memory_retrieval_recently_used(step, cooldown=8)
    ):
        return emit(
            "retrieve_relevant_history",
            memory_query=_default_memory_query(obs),
            conf=max(0.42, confidence - 0.1),
        )

    if consented_ids and best_site and budget > 1400:
        return emit("allocate_to_site", patient_id=consented_ids[0], site_id=best_site)

    if recontact_ids and budget > 200:
        first_due = policy_state.patients.get(recontact_ids[0], {}).get("followup_due_step")
        if isinstance(first_due, int) and first_due <= step + 2:
            return emit("recontact", patient_id=recontact_ids[0], conf=max(confidence, 0.68))

    if regulatory_hold_days > 0:
        if recontact_ids and budget > 200:
            return emit("recontact", patient_id=recontact_ids[0], conf=max(confidence, 0.7))
        if (
            best_negotiate_site
            and budget > 400
            and delayed_effects_pending < 3
            and not policy_state.strategy_recently_used(
                f"negotiate_site_{_site_suffix(best_negotiate_site)}", step, cooldown=14
            )
        ):
            return emit(
                "adjust_strategy",
                strategy_change=f"negotiate_site_{_site_suffix(best_negotiate_site)}",
                conf=max(confidence, 0.65),
            )

    if (
        patient_uncertainty > 0.34
        and budget > 400
        and time_left > 12
        and not policy_state.strategy_recently_used("tighten_criteria", step, cooldown=14)
    ):
        return emit("adjust_strategy", strategy_change="tighten_criteria", conf=0.68)

    if available and budget > 900 and regulatory_hold_days <= 0:
        best_patient = max(
            available,
            key=lambda patient: float(patient.get("eligibility_score", 0.0))
            * (1.0 - float(patient.get("dropout_risk", 0.0))),
        )
        patient_id = best_patient.get("id")
        if patient_id:
            return emit("screen_patient", patient_id=patient_id, conf=max(confidence, 0.64))

    if recontact_ids and budget > 200:
        return emit("recontact", patient_id=recontact_ids[0], conf=max(confidence, 0.66))

    if (
        budget > 400
        and delayed_effects_pending < 2
        and obs.get("enrolled_so_far", 0) < max(1, obs.get("target_enrollment", 1)) * 0.2
        and patient_uncertainty < 0.28
        and not policy_state.strategy_recently_used("increase_outreach", step, cooldown=16)
    ):
        return emit("adjust_strategy", strategy_change="increase_outreach", conf=0.6)

    if time_left <= 2 or budget < 250:
        progress = obs.get("enrolled_so_far", 0) / max(1, obs.get("target_enrollment", 1))
        return emit("stop_recruitment", conf=min(0.95, max(0.1, progress)))

    return None


def rule_based_action(
    obs: dict, step: int = 0, policy_state: Optional[PolicyState] = None
) -> dict:
    """Heuristic policy: screen available patients, allocate consented ones, manage budget."""
    policy_state = policy_state or PolicyState()
    enrolled = obs.get("enrolled_so_far", 0)
    target = obs.get("target_enrollment", 100)
    budget = obs.get("budget_remaining", 0)
    time_left = obs.get("time_to_deadline_days", 180)
    available = obs.get("available_patients", [])
    uncertainty = obs.get("uncertainty_level", 0)
    screening_backlog = obs.get("screening_backlog", 0)
    milestones = obs.get("milestones", {})
    constraints = obs.get("active_constraints", {})
    uncertainty_components = obs.get("uncertainty_components", {})

    patient_uncertainty = float(uncertainty_components.get("patient_pool", uncertainty))
    site_uncertainty = float(uncertainty_components.get("site_operations", 0.0))
    policy_uncertainty = float(uncertainty_components.get("policy", 0.0))
    regulatory_hold_days = int(constraints.get("regulatory_hold_days", 0))
    sponsor_pressure = bool(constraints.get("sponsor_pressure", False))
    site_bottleneck = bool(constraints.get("site_bottleneck", False))
    delayed_effects_pending = int(obs.get("delayed_effects_pending", 0))
    focused_site = constraints.get("focused_site", "")
    current_plan = obs.get("current_plan", {})
    indexed_memory_summary = obs.get("indexed_memory_summary", {})
    retrieved_memory_context = obs.get("retrieved_memory_context", "")
    milestone_potential = float(obs.get("milestone_potential", 0.0) or 0.0)
    active_milestone = obs.get("active_milestone", "")

    def emit(
        action_type: str,
        patient_id: Optional[str] = None,
        site_id: Optional[str] = None,
        strategy_change: Optional[str] = None,
        confidence_override: Optional[float] = None,
        plan_id: Optional[str] = None,
        plan_summary: Optional[str] = None,
        target_phase: Optional[str] = None,
        memory_key: Optional[str] = None,
        memory_query: Optional[str] = None,
        memory_payload: Optional[str] = None,
    ) -> dict:
        return {
            "action_type": action_type,
            "patient_id": patient_id,
            "site_id": site_id,
            "strategy_change": strategy_change,
            "hypothesis": hypothesis,
            "confidence": confidence_override if confidence_override is not None else confidence,
            "plan_id": plan_id,
            "plan_summary": plan_summary,
            "target_phase": target_phase,
            "memory_key": memory_key,
            "memory_query": memory_query,
            "memory_payload": memory_payload,
            "token_cost": _estimated_token_cost(action_type),
        }

    # Infer hypothesis and confidence
    hypothesis = _infer_hypothesis(obs)
    confidence = _infer_confidence(obs, step)

    if obs.get("difficulty") == 3:
        hard_action = _hard_mode_action(obs, step, policy_state, hypothesis, confidence)
        if hard_action is not None:
            return hard_action

    consented_ids = policy_state.consented_pending_ids(step)
    recontact_ids = policy_state.recontact_candidate_ids(step)
    best_allocate_site = _best_site(obs, mode="allocate")
    best_negotiate_site = _best_site(obs, mode="negotiate")
    best_allocate_info = obs.get("site_performance", {}).get(best_allocate_site or "", {})
    negotiate_strategy = (
        f"negotiate_site_{_site_suffix(best_negotiate_site)}"
        if best_negotiate_site
        else None
    )
    focus_strategy = (
        f"focus_site_{_site_suffix(best_allocate_site)}" if best_allocate_site else None
    )
    recommended_phase = _recommended_phase(obs)
    plan_phase = current_plan.get("target_phase") if isinstance(current_plan, dict) else None

    if (
        not plan_phase or plan_phase != recommended_phase
    ) and not policy_state.planning_recently_used(step):
        return emit(
            "plan_next_phase",
            target_phase=recommended_phase,
            plan_id=f"plan-{step}-{recommended_phase}",
            plan_summary=_plan_summary_for_phase(recommended_phase, obs),
            confidence_override=max(confidence, min(0.85, 0.45 + milestone_potential * 0.4)),
        )

    if (
        indexed_memory_summary.get("entries", 0) <= 0
        or (
            active_milestone
            and indexed_memory_summary.get(recommended_phase, 0) <= 0
            and not policy_state.memory_write_recently_used(step)
        )
    ):
        return emit(
            "summarize_and_index",
            memory_key=_default_memory_key(obs),
            memory_payload=_default_memory_payload(obs),
            confidence_override=max(0.45, confidence - 0.05),
        )

    if (
        indexed_memory_summary.get("entries", 0) > 0
        and not retrieved_memory_context
        and milestone_potential < 0.72
        and not policy_state.memory_retrieval_recently_used(step)
    ):
        return emit(
            "retrieve_relevant_history",
            memory_query=_default_memory_query(obs),
            confidence_override=max(0.4, confidence - 0.08),
        )

    # If we have consented patients not yet enrolled, allocate them first.
    if consented_ids and best_allocate_site and budget > 1500:
        first_due = policy_state.patients.get(consented_ids[0], {}).get("followup_due_step")
        if (
            isinstance(first_due, int)
            and first_due <= step + 1
            and float(best_allocate_info.get("avg_wait_days", 0.0)) >= 5.5
        ):
            return emit("recontact", patient_id=consented_ids[0])
        return emit(
            "allocate_to_site",
            patient_id=consented_ids[0],
            site_id=best_allocate_site,
        )

    # If screening is blocked, spend the step on follow-up or site preparation.
    if regulatory_hold_days > 0:
        if recontact_ids and budget > 200:
            return emit("recontact", patient_id=recontact_ids[0])
        if (
            negotiate_strategy
            and budget > 400
            and delayed_effects_pending < 3
            and not policy_state.strategy_recently_used(negotiate_strategy, step, cooldown=12)
        ):
            return emit("adjust_strategy", strategy_change=negotiate_strategy)

    # Rescue follow-up windows before delayed effects close them.
    if recontact_ids and budget > 200:
        first_due = policy_state.patients.get(recontact_ids[0], {}).get(
            "followup_due_step"
        )
        if isinstance(first_due, int) and first_due <= step + 1:
            return emit("recontact", patient_id=recontact_ids[0])

    # If site operations are the main bottleneck, negotiate capacity/wait improvements.
    if (
        negotiate_strategy
        and budget > 400
        and (site_bottleneck or site_uncertainty > 0.45)
        and delayed_effects_pending < 3
        and not policy_state.strategy_recently_used(negotiate_strategy, step, cooldown=12)
    ):
        return emit("adjust_strategy", strategy_change=negotiate_strategy)

    # Focus a strong site once site bias is apparent.
    if (
        focus_strategy
        and hypothesis == "site_bias"
        and focused_site != best_allocate_site
        and time_left > 25
        and budget > 400
        and not policy_state.strategy_recently_used(focus_strategy, step, cooldown=12)
    ):
        return emit("adjust_strategy", strategy_change=focus_strategy)

    # If the patient pool has drifted noisy, tighten before screening more.
    if (
        patient_uncertainty > 0.55
        and budget > 400
        and time_left > 25
        and not policy_state.strategy_recently_used("tighten_criteria", step)
    ):
        return emit("adjust_strategy", strategy_change="tighten_criteria")

    # If behind schedule, push growth only when we are not already over-noisy.
    expected_progress = 1.0 - (time_left / 180.0)
    actual_progress = enrolled / max(1, target)
    if sponsor_pressure or actual_progress < expected_progress * 0.75:
        if (
            budget > 400
            and time_left > 20
            and delayed_effects_pending < 3
            and not policy_state.strategy_recently_used("increase_outreach", step, cooldown=10)
        ):
            return emit("adjust_strategy", strategy_change="increase_outreach")
        if (
            patient_uncertainty < 0.4
            and budget > 400
            and time_left > 20
            and not policy_state.strategy_recently_used("relax_criteria", step, cooldown=10)
        ):
            return emit("adjust_strategy", strategy_change="relax_criteria")

    # When backlog grows, convert known candidates before screening more.
    if (screening_backlog > 3 or delayed_effects_pending > 2) and recontact_ids and budget > 200:
        return emit("recontact", patient_id=recontact_ids[0])

    # Default: screen next available patient (MUST provide patient_id)
    if available and budget > 900 and regulatory_hold_days <= 0:
        patient_id = _best_available_patient(obs)
        if patient_id:
            return emit("screen_patient", patient_id=patient_id)

    if recontact_ids and budget > 200:
        return emit("recontact", patient_id=recontact_ids[0])

    # If near target or low budget, stop with calibrated confidence
    if enrolled >= target or budget < 500 or (time_left <= 5 and milestones.get("75pct")):
        return emit(
            "stop_recruitment",
            confidence_override=min(0.95, enrolled / max(1, target)),
        )

    # No patients available but budget left: keep the episode alive with a non-redundant strategy step.
    if (
        negotiate_strategy
        and budget > 400
        and delayed_effects_pending < 3
        and not policy_state.strategy_recently_used(negotiate_strategy, step, cooldown=12)
    ):
        return emit("adjust_strategy", strategy_change=negotiate_strategy)
    if (
        budget > 400
        and time_left > 10
        and policy_uncertainty < 0.55
        and delayed_effects_pending < 3
        and not policy_state.strategy_recently_used("increase_outreach", step, cooldown=10)
    ):
        return emit("adjust_strategy", strategy_change="increase_outreach")

    if available and regulatory_hold_days <= 0 and budget > 900:
        patient_id = _best_available_patient(obs)
        if patient_id:
            return emit("screen_patient", patient_id=patient_id)

    # Fallback: only stop when late or resource-constrained.
    if time_left <= 3 or budget < 250:
        return emit(
            "stop_recruitment",
            confidence_override=min(0.95, enrolled / max(1, target)),
        )

    if budget > 200 and recontact_ids:
        return emit("recontact", patient_id=recontact_ids[0])

    return emit(
        "adjust_strategy",
        strategy_change="increase_outreach",
    )


# -- LLM-based policy --
SYSTEM_PROMPT = """You are an expert Clinical Trial Recruitment Optimizer at a top-5 pharma company.

Your goal is to maximize successful enrollment while minimizing cost, timeline slippage, and dropout.

CURRENT STATE:
- Day: {timestamp} / 180
- Budget left: ${budget_remaining:.0f}
- Enrolled: {enrolled_so_far}/{target_enrollment}
- Funnel: {current_funnel}
- Time to deadline: {time_to_deadline_days} days
- Uncertainty: {uncertainty_level:.2f}
- 7-day dropout rate: {dropout_rate_7d:.2f}
- Screening backlog: {screening_backlog}
- Recent events: {recent_events}
- Causal insight: {causal_insight}
- Last hypothesis accuracy: {hypothesis_accuracy:.2f}

AVAILABLE PATIENTS (up to 5):
{patient_summary}

TRACKED ACTIONABLE PATIENTS:
{tracked_patient_summary}

SITE PERFORMANCE:
{site_summary}

LONG-HORIZON SIGNALS:
- Milestones reached: {milestones}
- Active constraints: {active_constraints}
- Uncertainty components: {uncertainty_components}
- Patient memory summary: {patient_memory_summary}
- Current plan: {current_plan}
- Indexed memory summary: {indexed_memory_summary}
- Retrieved memory context: {retrieved_memory_context}
- Milestone potential: {milestone_potential:.2f}
- Active milestone frontier: {active_milestone}
- Token budget remaining: {token_budget_remaining}
- Token usage so far: {token_usage_so_far}
- Token efficiency score: {token_efficiency_score:.2f}
- Delayed effects pending: {delayed_effects_pending}
- Counterfactual hint: {counterfactual_hint}

AVAILABLE ACTIONS:
1. screen_patient (patient_id) - run screening ($800-900, may find eligible)
2. recontact (patient_id) - re-engage dropped interest (low cost, uncertain)
3. allocate_to_site (patient_id, site_id) - assign consented patient to site for enrollment
4. adjust_strategy (strategy_change) - one of: {strategy_options}
5. plan_next_phase (target_phase, plan_id, plan_summary) - set an explicit high-level next phase
6. summarize_and_index (memory_key, memory_payload) - store a compact memory entry for later retrieval
7. retrieve_relevant_history (memory_query) - retrieve indexed history relevant to the current bottleneck
8. stop_recruitment - end episode early (only when confident enrollment targets are met or unachievable)

REASONING REQUIREMENTS:
You MUST include a hypothesis about what is driving trial dynamics:
- "dropout_dominant" - dropout is the main challenge
- "noise_dominant" - uncertainty/noise is the main challenge
- "site_bias" - site performance variance is the main challenge
- "confounding" - multiple factors interacting

You MUST include a confidence score (0.0-1.0) for your hypothesis.
IMPORTANT: Be CONSISTENT with your hypothesis. Switching hypotheses too often is penalized.
IMPORTANT: When stopping recruitment, calibrate your confidence to match actual enrollment progress.
IMPORTANT: Prefer lower-token actions when they achieve similar progress. Large planning/memory actions should pay for themselves.

Respond with EXACTLY one JSON object:
{{"action_type": "<action>", "patient_id": "<id or null>", "site_id": "<id or null>", "strategy_change": "<change or null>", "plan_id": "<id or null>", "plan_summary": "<summary or null>", "target_phase": "<phase or null>", "memory_key": "<key or null>", "memory_query": "<query or null>", "memory_payload": "<payload or null>", "token_cost": <int or null>, "hypothesis": "<hypothesis>", "confidence": <float>}}

No markdown, no explanation."""


def llm_action(
    client: OpenAI, obs: dict, step_num: int, policy_state: PolicyState
) -> dict:
    # Format patient summary
    patients = obs.get("available_patients", [])
    if patients:
        patient_lines = []
        for p in patients[:5]:
            patient_lines.append(
                f"  {p.get('id')}: age={p.get('age')}, elig={p.get('eligibility_score', 0):.2f}, "
                f"dropout_risk={p.get('dropout_risk', 0):.2f}"
            )
        patient_summary = "\n".join(patient_lines)
    else:
        patient_summary = "  (none available)"

    # Format site summary
    sites = obs.get("site_performance", {})
    if sites:
        site_lines = []
        for sid, sinfo in sites.items():
            site_lines.append(
                f"  {sid}: conv={sinfo.get('conversion_rate', 0):.2f}, "
                f"wait={sinfo.get('avg_wait_days', 0):.1f}d, "
                f"capacity={sinfo.get('capacity_remaining', 0)}"
            )
        site_summary = "\n".join(site_lines)
    else:
        site_summary = "  (no sites)"

    tracked_patient_summary = policy_state.describe_tracked_patients(step_num)
    strategy_options = ", ".join(_valid_strategy_choices(obs))

    prompt = SYSTEM_PROMPT.format(
        timestamp=obs.get("timestamp", 0),
        budget_remaining=obs.get("budget_remaining", 0),
        enrolled_so_far=obs.get("enrolled_so_far", 0),
        target_enrollment=obs.get("target_enrollment", 100),
        current_funnel=obs.get("current_funnel", {}),
        time_to_deadline_days=obs.get("time_to_deadline_days", 180),
        uncertainty_level=obs.get("uncertainty_level", 0),
        dropout_rate_7d=obs.get("dropout_rate_7d", 0),
        screening_backlog=obs.get("screening_backlog", 0),
        recent_events=obs.get("recent_events", []),
        causal_insight=obs.get("causal_insight", "No insight yet."),
        hypothesis_accuracy=obs.get("hypothesis_accuracy", 0),
        patient_summary=patient_summary,
        tracked_patient_summary=tracked_patient_summary,
        site_summary=site_summary,
        strategy_options=strategy_options,
        milestones=obs.get("milestones", {}),
        active_constraints=obs.get("active_constraints", {}),
        uncertainty_components=obs.get("uncertainty_components", {}),
        patient_memory_summary=obs.get("patient_memory_summary", {}),
        current_plan=obs.get("current_plan", {}),
        indexed_memory_summary=obs.get("indexed_memory_summary", {}),
        retrieved_memory_context=obs.get("retrieved_memory_context", ""),
        milestone_potential=float(obs.get("milestone_potential", 0.0) or 0.0),
        active_milestone=obs.get("active_milestone", ""),
        token_budget_remaining=int(obs.get("token_budget_remaining", 0) or 0),
        token_usage_so_far=int(obs.get("token_usage_so_far", 0) or 0),
        token_efficiency_score=float(obs.get("token_efficiency_score", 1.0) or 0.0),
        delayed_effects_pending=obs.get("delayed_effects_pending", 0),
        counterfactual_hint=obs.get("counterfactual_hint", ""),
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=TEMPERATURE,
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        action = json.loads(text)

        normalized = _normalize_action(action, obs, step_num, policy_state)
        if normalized is None:
            return rule_based_action(obs, step_num, policy_state)
        return normalized
    except Exception:
        return rule_based_action(obs, step_num, policy_state)


# -- Run one task --
def run_task(task_id: str, client: OpenAI) -> float:
    env = EnvClient(ENV_URL)
    policy_state = PolicyState()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_info = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id)
        obs = result["observation"]
        last_info = result.get("info", {})
        policy_state.reset(obs)

        while not result.get("done", False):
            prev_obs = obs
            if steps_taken % LLM_CALL_INTERVAL == 0:
                action = llm_action(client, obs, steps_taken, policy_state)
            else:
                action = rule_based_action(obs, steps_taken, policy_state)

            action = _normalize_action(action, obs, steps_taken, policy_state)
            if action is None:
                action = rule_based_action(obs, steps_taken, policy_state)
                action = _normalize_action(action, obs, steps_taken, policy_state)
            if action is None:
                action = {
                    "action_type": "stop_recruitment",
                    "patient_id": None,
                    "site_id": None,
                    "strategy_change": None,
                    "plan_id": None,
                    "plan_summary": None,
                    "target_phase": None,
                    "memory_key": None,
                    "memory_query": None,
                    "memory_payload": None,
                    "token_cost": _estimated_token_cost("stop_recruitment"),
                    "hypothesis": _infer_hypothesis(obs),
                    "confidence": min(
                        0.95,
                        obs.get("enrolled_so_far", 0)
                        / max(1, obs.get("target_enrollment", 1)),
                    ),
                }

            action_str = _format_action(action)

            try:
                result = env.step(action)
                obs = result["observation"]
                policy_state.update(prev_obs, action, result, steps_taken)
                reward = float(result.get("reward", 0.0) or 0.0)
                done = bool(result.get("done", False))
                last_info = result.get("info", {})
                error = last_info.get("last_action_error")
                rewards.append(reward)
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )
            except Exception as exc:
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

        final_score = last_info.get("final_score")
        if isinstance(final_score, (int, float)):
            score = float(final_score)
        else:
            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(0.999, max(0.001, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        env.close()
        score = min(0.999, max(0.001, score))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# -- Main --
def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN environment variable is required.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["easy_bench", "medium_bench", "hard_bench"]
    for task_id in tasks:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
