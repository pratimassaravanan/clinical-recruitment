"""Deterministic patient pool and event traces for Clinical Recruitment environment.

All traces are pre-computed with fixed seeds for reproducibility.
Each trace covers 180 simulated steps of a clinical trial recruitment period.
"""

import copy
import random
import re
from typing import List, Dict, Any, Optional


PUBLIC_TASKS = ("easy_bench", "medium_bench", "hard_bench")
PROGRESSIVE_HORIZONS = (30, 90, 180)
_STAGE_TASK_PATTERN = re.compile(
    r"^(easy_bench|medium_bench|hard_bench)_stage_(30|90|180)$"
)

PUBLIC_TASK_INFO = {
    "easy_bench": {
        "name": "Basic Eligibility Screening",
        "description": "Stable patient pool, low dropout, generous budget/time. Goal: reach 80% enrollment with high screening accuracy.",
        "difficulty": "easy",
    },
    "medium_bench": {
        "name": "Full Funnel with Site Allocation",
        "description": "Moderate uncertainty, 3 sites with different performance, some dropout risk. Optimize across sites.",
        "difficulty": "medium",
    },
    "hard_bench": {
        "name": "Multi-Objective Pipeline Under Pressure",
        "description": "Tight budget/time, high dropout, non-stationary patient quality, curriculum injections. Multi-objective optimization.",
        "difficulty": "hard",
    },
}


def _generate_patient_pool(
    rng: random.Random, count: int, noise: float
) -> List[Dict[str, Any]]:
    """Generate a pool of candidate patients with eligibility and dropout characteristics."""
    patients = []
    for i in range(count):
        age = rng.randint(18, 80)
        # Eligibility score: higher = more likely to pass screening
        base_elig = rng.uniform(0.3, 0.95)
        elig_noise = rng.gauss(0, noise)
        eligibility_score = max(0.05, min(0.99, base_elig + elig_noise))
        # Dropout risk: higher = more likely to drop out after enrollment
        base_dropout = rng.uniform(0.05, 0.55)
        dropout_noise = rng.gauss(0, noise * 0.5)
        dropout_risk = max(0.01, min(0.90, base_dropout + dropout_noise))
        # Willingness: affects recontact success
        willingness = rng.uniform(0.2, 0.9)
        patients.append(
            {
                "id": f"P-{1000 + i}",
                "age": age,
                "eligibility_score": round(eligibility_score, 3),
                "dropout_risk": round(dropout_risk, 3),
                "willingness": round(willingness, 3),
                "contacted": False,
                "screened": False,
                "eligible": False,
                "consented": False,
                "enrolled": False,
                "dropped": False,
                "site_assigned": None,
            }
        )
    return patients


def _generate_sites(
    rng: random.Random, num_sites: int, variance: float
) -> Dict[str, Dict[str, float]]:
    """Generate recruitment sites with varying performance characteristics."""
    sites = {}
    base_conversions = [0.70, 0.55, 0.40, 0.60, 0.50]
    base_waits = [5.0, 3.0, 8.0, 4.0, 6.0]
    base_capacities = [40, 30, 50, 35, 25]
    for i in range(num_sites):
        sid = chr(65 + i)  # A, B, C, ...
        conv = max(0.15, min(0.95, base_conversions[i % 5] + rng.gauss(0, variance)))
        wait = max(1.0, base_waits[i % 5] + rng.gauss(0, variance * 3))
        cap = max(10, int(base_capacities[i % 5] + rng.gauss(0, variance * 10)))
        sites[f"site_{sid}"] = {
            "conversion_rate": round(conv, 3),
            "avg_wait_days": round(wait, 1),
            "capacity_total": cap,
            "capacity_remaining": cap,
            "retention_rate": round(
                max(0.5, min(0.98, 0.85 + rng.gauss(0, variance))), 3
            ),
        }
    return sites


def _generate_events(
    rng: random.Random, num_days: int, event_density: float
) -> Dict[int, List[str]]:
    """Generate day-indexed event lists (dropouts, site delays, regulatory holds, etc.)."""
    events: Dict[int, List[str]] = {}
    event_types = [
        "patient_dropout",
        "site_delay",
        "regulatory_hold",
        "site_audit",
        "irb_delay",
        "consent_form_revision",
        "new_competitor_trial",
        "patient_complaint",
        "screening_backlog",
        "site_capacity_reduced",
        "protocol_amendment",
        "seasonal_slowdown",
    ]
    for day in range(num_days):
        day_events = []
        for etype in event_types:
            if rng.random() < event_density:
                day_events.append(etype)
        if day_events:
            events[day] = day_events
    return events


def generate_easy_trace() -> Dict[str, Any]:
    """Easy: stable patient pool, low dropout, generous budget/time, 1 site.
    Goal: reach 80% enrollment with high screening accuracy.
    """
    rng = random.Random(42)
    patients = _generate_patient_pool(rng, count=200, noise=0.05)
    sites = _generate_sites(rng, num_sites=1, variance=0.02)
    events = _generate_events(rng, num_days=180, event_density=0.02)
    # Curriculum injections: none for easy
    curriculum = []
    return {
        "patients": patients,
        "sites": sites,
        "events": events,
        "curriculum": curriculum,
        "budget": 120000.0,
        "target_enrollment": 80,
        "deadline_days": 180,
        "screening_cost": 600.0,
        "enrollment_cost": 1200.0,
        "recontact_cost": 100.0,
        "strategy_cost": 200.0,
        "dropout_base_rate": 0.05,
        "uncertainty_growth": 0.001,
        "world_type": "noise",  # easy: noise is the dominant dynamic
    }


def generate_medium_trace() -> Dict[str, Any]:
    """Medium: moderate uncertainty, 3 sites with different performance, some dropout."""
    rng = random.Random(123)
    patients = _generate_patient_pool(rng, count=350, noise=0.12)
    sites = _generate_sites(rng, num_sites=3, variance=0.08)
    events = _generate_events(rng, num_days=180, event_density=0.06)
    curriculum = []
    return {
        "patients": patients,
        "sites": sites,
        "events": events,
        "curriculum": curriculum,
        "budget": 150000.0,
        "target_enrollment": 120,
        "deadline_days": 180,
        "screening_cost": 800.0,
        "enrollment_cost": 1400.0,
        "recontact_cost": 150.0,
        "strategy_cost": 300.0,
        "dropout_base_rate": 0.10,
        "uncertainty_growth": 0.003,
        "world_type": "site_bias",  # medium: site variability is the dominant dynamic
    }


def generate_hard_trace() -> Dict[str, Any]:
    """Hard: tight budget/time, high dropout, non-stationary quality, curriculum injections."""
    rng = random.Random(777)
    patients = _generate_patient_pool(rng, count=500, noise=0.20)
    sites = _generate_sites(rng, num_sites=5, variance=0.15)
    events = _generate_events(rng, num_days=180, event_density=0.12)
    # Curriculum injections: periodic easy-pool resets
    curriculum = [
        {"start_day": 25, "duration": 8, "type": "easy_pool_reset"},
        {"start_day": 60, "duration": 10, "type": "easy_pool_reset"},
        {"start_day": 100, "duration": 7, "type": "easy_pool_reset"},
        {"start_day": 140, "duration": 12, "type": "easy_pool_reset"},
    ]
    return {
        "patients": patients,
        "sites": sites,
        "events": events,
        "curriculum": curriculum,
        "budget": 100000.0,
        "target_enrollment": 150,
        "deadline_days": 180,
        "screening_cost": 900.0,
        "enrollment_cost": 1500.0,
        "recontact_cost": 200.0,
        "strategy_cost": 400.0,
        "dropout_base_rate": 0.18,
        "uncertainty_growth": 0.005,
        "world_type": "dropout",  # hard: dropout is the dominant dynamic
    }


BASE_TASK_TRACES = {
    "easy_bench": generate_easy_trace(),
    "medium_bench": generate_medium_trace(),
    "hard_bench": generate_hard_trace(),
}

TASK_TRACES = BASE_TASK_TRACES


def make_stage_task_id(base_task_id: str, horizon_days: int) -> str:
    if base_task_id not in BASE_TASK_TRACES:
        raise KeyError(f"Unknown base task: {base_task_id}")
    if horizon_days not in PROGRESSIVE_HORIZONS:
        raise ValueError(f"Unsupported progressive horizon: {horizon_days}")
    return f"{base_task_id}_stage_{horizon_days}"


def resolve_base_task_id(task_id: str) -> str:
    if task_id in BASE_TASK_TRACES:
        return task_id
    match = _STAGE_TASK_PATTERN.match(task_id or "")
    if match:
        return match.group(1)
    return task_id


def get_stage_horizon_days(task_id: str) -> Optional[int]:
    match = _STAGE_TASK_PATTERN.match(task_id or "")
    if not match:
        return None
    return int(match.group(2))


def is_known_task(task_id: str) -> bool:
    return task_id in BASE_TASK_TRACES or _STAGE_TASK_PATTERN.match(task_id or "") is not None


def list_progressive_stage_tasks(base_task_id: Optional[str] = None) -> List[str]:
    task_ids: List[str] = []
    base_task_ids = [base_task_id] if base_task_id else list(PUBLIC_TASKS)
    for base_id in base_task_ids:
        for horizon_days in PROGRESSIVE_HORIZONS:
            task_ids.append(make_stage_task_id(base_id, horizon_days))
    return task_ids


def get_public_task_metadata() -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for task_id in PUBLIC_TASKS:
        trace = get_task_trace(task_id)
        metadata[task_id] = {
            **PUBLIC_TASK_INFO[task_id],
            "max_steps": int(trace.get("max_steps", trace.get("deadline_days", 180))),
            "initial_budget": round(float(trace.get("budget", 0.0)), 2),
            "target_enrollment": int(trace.get("target_enrollment", 0)),
        }
    return metadata


def _clip_curriculum(curriculum: List[Dict[str, Any]], horizon_days: int) -> List[Dict[str, Any]]:
    clipped: List[Dict[str, Any]] = []
    for injection in curriculum:
        start_day = int(injection.get("start_day", 0))
        end_day = start_day + int(injection.get("duration", 0))
        if start_day >= horizon_days:
            continue
        clipped_end_day = min(end_day, horizon_days - 1)
        clipped_injection = dict(injection)
        clipped_injection["duration"] = max(0, clipped_end_day - start_day)
        clipped.append(clipped_injection)
    return clipped


def build_progressive_trace(base_task_id: str, horizon_days: int) -> Dict[str, Any]:
    if base_task_id not in BASE_TASK_TRACES:
        raise KeyError(f"Unknown base task: {base_task_id}")
    if horizon_days not in PROGRESSIVE_HORIZONS:
        raise ValueError(f"Unsupported progressive horizon: {horizon_days}")

    trace = copy.deepcopy(BASE_TASK_TRACES[base_task_id])
    full_horizon = int(trace.get("deadline_days", 180))
    if horizon_days >= full_horizon:
        trace["task_id"] = make_stage_task_id(base_task_id, full_horizon)
        trace["stage_of"] = base_task_id
        trace["progressive_stage_days"] = full_horizon
        trace["max_steps"] = full_horizon
        return trace

    ratio = horizon_days / max(1, full_horizon)
    trace["task_id"] = make_stage_task_id(base_task_id, horizon_days)
    trace["stage_of"] = base_task_id
    trace["progressive_stage_days"] = horizon_days
    trace["deadline_days"] = horizon_days
    trace["max_steps"] = horizon_days
    trace["target_enrollment"] = max(
        1,
        int(round(int(trace.get("target_enrollment", 1)) * ratio)),
    )
    trace["budget"] = round(float(trace.get("budget", 0.0)) * (0.35 + 0.65 * ratio), 2)
    trace["events"] = {
        day: list(events)
        for day, events in trace.get("events", {}).items()
        if int(day) < horizon_days
    }
    trace["curriculum"] = _clip_curriculum(trace.get("curriculum", []), horizon_days)
    return trace


def get_task_trace(task_id: str) -> Dict[str, Any]:
    if task_id in BASE_TASK_TRACES:
        return copy.deepcopy(BASE_TASK_TRACES[task_id])
    match = _STAGE_TASK_PATTERN.match(task_id or "")
    if match:
        return build_progressive_trace(match.group(1), int(match.group(2)))
    raise KeyError(f"Unknown task: {task_id}")
