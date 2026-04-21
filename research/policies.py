"""Offline research policies for adaptive clinical recruitment experiments."""

from __future__ import annotations

from typing import Dict, Any, Optional

from inference import PolicyState, rule_based_action, _normalize_action


class ResearchPolicy:
    """Small wrapper around a policy + persistent state for experiment runs."""

    def __init__(self, name: str):
        self.name = name
        self.state = PolicyState()

    def reset(self, initial_obs: Dict[str, Any]) -> None:
        self.state.reset(initial_obs)

    def act(self, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        raise NotImplementedError

    def update(
        self,
        prev_obs: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any],
        step_num: int,
    ) -> None:
        self.state.update(prev_obs, action, result, step_num)


class RuleBasedPolicy(ResearchPolicy):
    def __init__(self):
        super().__init__(name="rule_based_memory")

    def act(self, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        action = rule_based_action(obs, step_num, self.state)
        normalized = _normalize_action(action, obs, step_num, self.state)
        if normalized is None:
            raise RuntimeError(f"Failed to normalize rule_based action at step {step_num}")
        return normalized


class GreedyScreenPolicy(ResearchPolicy):
    """Strong short-horizon baseline: screen aggressively, rarely adapts."""

    def __init__(self):
        super().__init__(name="greedy_screen")

    def act(self, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        available = obs.get("available_patients", [])
        budget = obs.get("budget_remaining", 0)
        target = obs.get("target_enrollment", 1)
        enrolled = obs.get("enrolled_so_far", 0)
        best_patient_id: Optional[str] = None
        if available:
            best_patient = max(
                available,
                key=lambda patient: float(patient.get("eligibility_score", 0.0))
                * (1.0 - float(patient.get("dropout_risk", 0.0)) * 0.6),
            )
            best_patient_id = best_patient.get("id")

        action = {
            "action_type": "screen_patient" if best_patient_id and budget > 900 else "stop_recruitment",
            "patient_id": best_patient_id,
            "site_id": None,
            "strategy_change": None,
            "hypothesis": "noise_dominant",
            "confidence": min(0.9, max(0.2, enrolled / max(1, target))),
        }
        normalized = _normalize_action(action, obs, step_num, self.state)
        if normalized is None:
            return {
                "action_type": "stop_recruitment",
                "patient_id": None,
                "site_id": None,
                "strategy_change": None,
                "hypothesis": "noise_dominant",
                "confidence": min(0.9, max(0.2, enrolled / max(1, target))),
            }
        return normalized


class ConservativeRetentionPolicy(ResearchPolicy):
    """Long-horizon baseline: tighter criteria and more follow-up / conversion focus."""

    def __init__(self):
        super().__init__(name="conservative_retention")

    def act(self, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        budget = obs.get("budget_remaining", 0)
        constraints = obs.get("active_constraints", {})
        available = obs.get("available_patients", [])
        recontact_ids = self.state.recontact_candidate_ids(step_num)
        consented_ids = self.state.consented_pending_ids(step_num)

        if consented_ids and budget > 1400:
            sites = obs.get("site_performance", {})
            site_id = None
            best_score = -1.0
            for sid, sinfo in sites.items():
                capacity = float(sinfo.get("capacity_remaining", 0))
                if capacity <= 0:
                    continue
                score = (
                    float(sinfo.get("conversion_rate", 0.0))
                    * float(sinfo.get("retention_rate", 0.8))
                    / max(1.0, float(sinfo.get("avg_wait_days", 0.0)))
                )
                if score > best_score:
                    best_score = score
                    site_id = sid
            action = {
                "action_type": "allocate_to_site",
                "patient_id": consented_ids[0],
                "site_id": site_id,
                "strategy_change": None,
                "hypothesis": "dropout_dominant",
                "confidence": 0.7,
            }
        elif recontact_ids and budget > 200:
            action = {
                "action_type": "recontact",
                "patient_id": recontact_ids[0],
                "site_id": None,
                "strategy_change": None,
                "hypothesis": "dropout_dominant",
                "confidence": 0.72,
            }
        elif constraints.get("regulatory_hold_days", 0) <= 0 and available and budget > 900:
            best_patient = max(
                available,
                key=lambda patient: float(patient.get("eligibility_score", 0.0))
                * (1.0 - float(patient.get("dropout_risk", 0.0))),
            )
            action = {
                "action_type": "screen_patient",
                "patient_id": best_patient.get("id"),
                "site_id": None,
                "strategy_change": None,
                "hypothesis": "dropout_dominant",
                "confidence": 0.66,
            }
        else:
            action = {
                "action_type": "adjust_strategy",
                "patient_id": None,
                "site_id": None,
                "strategy_change": "tighten_criteria",
                "hypothesis": "dropout_dominant",
                "confidence": 0.62,
            }

        normalized = _normalize_action(action, obs, step_num, self.state)
        if normalized is None:
            fallback = rule_based_action(obs, step_num, self.state)
            normalized = _normalize_action(fallback, obs, step_num, self.state)
        if normalized is None:
            raise RuntimeError(
                f"Failed to normalize conservative_retention action at step {step_num}"
            )
        return normalized


class SiteNegotiationPolicy(ResearchPolicy):
    """Site-centric policy that emphasizes negotiation and focused conversion."""

    def __init__(self):
        super().__init__(name="site_negotiation")

    def act(self, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        budget = obs.get("budget_remaining", 0)
        sites = obs.get("site_performance", {})
        available = obs.get("available_patients", [])
        consented_ids = self.state.consented_pending_ids(step_num)

        preferred_site = None
        preferred_suffix = ""
        best_score = -1.0
        for sid, sinfo in sites.items():
            capacity = float(sinfo.get("capacity_remaining", 0))
            if capacity <= 0:
                continue
            score = (
                float(sinfo.get("conversion_rate", 0.0)) * 2.0
                + float(sinfo.get("retention_rate", 0.8))
                - float(sinfo.get("avg_wait_days", 0.0)) * 0.08
            )
            if score > best_score:
                best_score = score
                preferred_site = sid
                preferred_suffix = sid.replace("site_", "")

        if (
            preferred_site
            and budget > 450
            and not self.state.strategy_recently_used(
                f"negotiate_site_{preferred_suffix}", step_num, cooldown=12
            )
        ):
            action = {
                "action_type": "adjust_strategy",
                "patient_id": None,
                "site_id": None,
                "strategy_change": f"negotiate_site_{preferred_suffix}",
                "hypothesis": "site_bias",
                "confidence": 0.74,
            }
        elif consented_ids and preferred_site and budget > 1400:
            action = {
                "action_type": "allocate_to_site",
                "patient_id": consented_ids[0],
                "site_id": preferred_site,
                "strategy_change": None,
                "hypothesis": "site_bias",
                "confidence": 0.76,
            }
        elif available and budget > 900:
            best_patient = max(
                available,
                key=lambda patient: float(patient.get("eligibility_score", 0.0))
                * (1.0 - float(patient.get("dropout_risk", 0.0)) * 0.7),
            )
            action = {
                "action_type": "screen_patient",
                "patient_id": best_patient.get("id"),
                "site_id": None,
                "strategy_change": None,
                "hypothesis": "site_bias",
                "confidence": 0.7,
            }
        else:
            action = {
                "action_type": "adjust_strategy",
                "patient_id": None,
                "site_id": None,
                "strategy_change": f"focus_site_{preferred_suffix}" if preferred_suffix else "increase_outreach",
                "hypothesis": "site_bias",
                "confidence": 0.68,
            }

        normalized = _normalize_action(action, obs, step_num, self.state)
        if normalized is None:
            fallback = rule_based_action(obs, step_num, self.state)
            normalized = _normalize_action(fallback, obs, step_num, self.state)
        if normalized is None:
            raise RuntimeError(
                f"Failed to normalize site_negotiation action at step {step_num}"
            )
        return normalized


POLICY_REGISTRY = {
    "rule_based_memory": RuleBasedPolicy,
    "greedy_screen": GreedyScreenPolicy,
    "conservative_retention": ConservativeRetentionPolicy,
    "site_negotiation": SiteNegotiationPolicy,
}
