"""Predictable skill summaries for abstract long-horizon control."""

from __future__ import annotations

from typing import Any, Dict, List


SKILL_LIBRARY = {
    "screening_push": {"phase": "screening", "expected_action": "screen_patient"},
    "conversion_rescue": {"phase": "conversion", "expected_action": "recontact"},
    "allocation_push": {"phase": "allocation", "expected_action": "allocate_to_site"},
    "recovery_protocol": {"phase": "recovery", "expected_action": "adjust_strategy"},
}


def infer_skills(obs: Dict[str, Any]) -> List[str]:
    memory = obs.get("patient_memory_summary", {})
    constraints = obs.get("active_constraints", {})
    skills: List[str] = []
    if int(memory.get("consented_pending_allocation", 0)) > 0:
        skills.append("allocation_push")
    if int(memory.get("followup_due", 0)) > 0:
        skills.append("conversion_rescue")
    if int(constraints.get("regulatory_hold_days", 0)) > 0:
        skills.append("recovery_protocol")
    if not skills:
        skills.append("screening_push")
    return skills
