"""Deterministic graders for Clinical Recruitment tasks. Each returns scores in (0, 1) with partial credit.

Grading includes hypothesis accuracy and reasoning consistency (Upgrades 1-4).
"""

from models import Observation


def _initial_budget(final_obs: Observation) -> float:
    return max(1.0, float(getattr(final_obs, "initial_budget", 0.0) or 0.0))


def _episode_horizon(final_obs: Observation, history: list) -> int:
    max_steps = int(getattr(final_obs, "max_steps", 0) or 0)
    if max_steps > 0:
        return max_steps
    remaining = int(getattr(final_obs, "time_to_deadline_days", 0) or 0)
    return max(1, len(history) + remaining)


def _hypothesis_consistency_score(history: list) -> float:
    """Score the agent's hypothesis consistency. Fewer switches = better."""
    hypotheses = [
        h.get("hypothesis", "unknown") for h in history if h.get("hypothesis")
    ]
    if len(hypotheses) < 2:
        return 1.0
    switches = sum(
        1 for i in range(1, len(hypotheses)) if hypotheses[i] != hypotheses[i - 1]
    )
    # 0-1 switches = perfect, 2-4 = partial, 5+ = poor
    if switches <= 1:
        return 1.0
    elif switches <= 4:
        return 1.0 - (switches - 1) * 0.2
    return 0.2


def _hypothesis_accuracy_score(history: list, world_type: str = "") -> float:
    """Score based on whether agent's hypothesis matches the ground truth world_type."""
    hypotheses = [
        h.get("hypothesis", "unknown") for h in history if h.get("hypothesis")
    ]
    if not hypotheses or not world_type:
        return 0.0

    # Map agent hypothesis names to world_type values
    hyp_to_world = {
        "dropout_dominant": "dropout",
        "noise_dominant": "noise",
        "site_bias": "site_bias",
        "confounding": "",  # no direct match
        "unknown": "",
    }

    # Score: what fraction of the last 20 steps had a correct hypothesis?
    window = hypotheses[-20:]
    correct = sum(1 for h in window if hyp_to_world.get(h, "") == world_type)
    return min(1.0, correct / max(1, len(window)))


def _plan_followthrough_score(history: list) -> float:
    plan_steps = [item for item in history if item.get("action") == "plan_next_phase"]
    if not plan_steps:
        return 0.0
    followthrough = sum(1 for item in history if item.get("plan_followthrough"))
    return min(1.0, followthrough / max(1, len(plan_steps) * 2))


def _memory_use_score(history: list) -> float:
    writes = sum(1 for item in history if item.get("action") == "summarize_and_index")
    hits = sum(1 for item in history if item.get("memory_hit"))
    if writes <= 0 and hits <= 0:
        return 0.0
    return min(1.0, (writes * 0.4 + hits * 0.6) / max(1.0, writes + 1.0))


def _milestone_potential_score(final_obs: Observation, history: list) -> float:
    deltas = [float(item.get("milestone_potential_delta", 0.0)) for item in history]
    positive = sum(max(0.0, delta) for delta in deltas)
    potential = float(getattr(final_obs, "milestone_potential", 0.0) or 0.0)
    return min(1.0, potential * 0.7 + min(0.3, positive * 0.2))


def _hindsight_alignment_score(final_obs: Observation, history: list) -> float:
    hindsight_available = bool(getattr(final_obs, "hindsight_available", False))
    if not hindsight_available:
        return 0.0
    followthrough = sum(1 for item in history if item.get("plan_followthrough"))
    memory_hits = sum(1 for item in history if item.get("memory_hit"))
    return min(1.0, 0.3 + followthrough * 0.02 + memory_hits * 0.03)


def _token_efficiency_grade(final_obs: Observation, history: list) -> float:
    efficiency = float(getattr(final_obs, "token_efficiency_score", 1.0) or 0.0)
    token_usage = int(getattr(final_obs, "token_usage_so_far", 0) or 0)
    useful_steps = sum(
        1
        for item in history
        if item.get("screen_success") or item.get("enrolled") or item.get("memory_hit")
    )
    if token_usage <= 0:
        return min(1.0, 0.4 + useful_steps * 0.02)
    density = useful_steps / max(1, len(history))
    return min(1.0, efficiency * 0.8 + density * 0.2)


def grade_easy_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade easy bench: stable patient pool, generous budget/time.

    Scoring (0.0-1.0), weights sum to 1.00:
      - Enrollment rate: 0.30 weight (% of target reached)
      - Budget efficiency: 0.17 weight (budget remaining / initial)
      - Screening accuracy: 0.17 weight (eligible / screened ratio)
      - Timeline: 0.10 weight (enrolled before deadline)
      - Hypothesis consistency: 0.10 weight (stable reasoning)
      - Hypothesis accuracy: 0.05 weight (correct world model)
      - Milestone potential: 0.03 weight
      - Plan follow-through: 0.02 weight
      - Token efficiency: 0.03 weight
      - Memory use: 0.03 weight
    """
    score = 0.0

    # Enrollment rate (30%) - how much of target was reached
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.30 * enrollment_pct

    # Budget efficiency (17%) - reward for not exhausting budget
    # BUT: only award if agent actually took productive actions (screened >= 10 patients)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / _initial_budget(final_obs)
    if screened >= 10:
        if budget_frac >= 0.20:
            score += 0.17
        elif budget_frac >= 0.0:
            score += 0.17 * (budget_frac / 0.20)
    # else: 0 — no free points for doing nothing

    # Screening accuracy (17%) - eligible / screened ratio
    eligible = funnel.get("eligible", 0)
    if screened > 0:
        accuracy = eligible / screened
        score += 0.17 * min(1.0, accuracy / 0.7)
    else:
        score += 0.0

    # Timeline (10%) - bonus for finishing early
    days_used = len(history)
    max_steps = _episode_horizon(final_obs, history)
    if enrolled >= target:
        time_frac = max(0.0, 1.0 - (days_used / max_steps))
        score += 0.10 * time_frac
    elif days_used < max_steps:
        score += 0.10 * (enrolled / max(1, target)) * 0.5

    # Hypothesis consistency (10%)
    score += 0.10 * _hypothesis_consistency_score(history)

    # Hypothesis accuracy (5%)
    score += 0.05 * _hypothesis_accuracy_score(history, final_obs.world_type)

    # Milestone potential / planning quality + memory use (8%)
    score += 0.03 * _milestone_potential_score(final_obs, history)
    score += 0.02 * _plan_followthrough_score(history)
    score += 0.03 * _memory_use_score(history)

    # Token efficiency (3%)
    score += 0.03 * _token_efficiency_grade(final_obs, history)

    return round(min(0.999, max(0.001, score)), 4)


def grade_medium_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade medium bench: multiple sites, moderate dropout, uncertainty.

    Scoring (0.0-1.0), weights sum to 1.00:
      - Enrollment rate: 0.25 weight
      - Retention: 0.15 weight (enrolled - dropped / enrolled)
      - Site utilization: 0.15 weight (used multiple sites effectively)
      - Budget efficiency: 0.10 weight
      - Hypothesis consistency: 0.10 weight
      - Hypothesis accuracy: 0.10 weight
      - Plan follow-through: 0.05 weight
      - Milestone potential: 0.05 weight
      - Token efficiency: 0.05 weight
    """
    score = 0.0

    # Enrollment rate (25%)
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.25 * enrollment_pct

    # Retention (15%) - low dropout
    funnel = final_obs.current_funnel
    total_enrolled_ever = funnel.get("enrolled", 0) + funnel.get("dropped", 0)
    dropped = funnel.get("dropped", 0)
    if total_enrolled_ever > 0:
        retention = 1.0 - (dropped / total_enrolled_ever)
        score += 0.15 * max(0.0, retention)
    else:
        score += 0.0

    # Site utilization (15%) - did agent use multiple sites?
    sites_used = set()
    for h in history:
        sid = h.get("site_id")
        if sid:
            sites_used.add(sid)
    if len(sites_used) >= 3:
        score += 0.15
    elif len(sites_used) == 2:
        score += 0.15 * 0.7
    elif len(sites_used) == 1:
        score += 0.15 * 0.3

    # Budget efficiency (10%) - only if agent engaged (screened >= 15)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / _initial_budget(final_obs)
    if screened >= 15:
        if budget_frac >= 0.15:
            score += 0.10
        elif budget_frac >= 0.0:
            score += 0.10 * (budget_frac / 0.15)

    # Hypothesis consistency (10%)
    score += 0.10 * _hypothesis_consistency_score(history)

    # Hypothesis accuracy (10%)
    score += 0.10 * _hypothesis_accuracy_score(history, final_obs.world_type)

    # Explicit planning + milestone shaping (10%)
    score += 0.05 * _plan_followthrough_score(history)
    score += 0.05 * _milestone_potential_score(final_obs, history)

    # Token efficiency (5%)
    score += 0.05 * _token_efficiency_grade(final_obs, history)

    return round(min(0.999, max(0.001, score)), 4)


def grade_hard_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade hard bench: tight budget, high dropout, curriculum injections.

    Scoring (0.0-1.0), weights sum to 1.00:
      - Enrollment rate: 0.15 weight
      - Retention: 0.10 weight
      - Budget efficiency: 0.10 weight
      - Dropout recovery: 0.10 weight (bounced back after high-dropout events)
      - Curriculum response: 0.10 weight (exploited easy-pool resets)
      - Strategy adaptation: 0.10 weight (used strategy changes effectively)
      - Hypothesis consistency: 0.10 weight (stable reasoning under chaos)
      - Hypothesis accuracy: 0.10 weight (identified dropout as dominant)
      - Memory use: 0.04 weight
      - Milestone potential: 0.03 weight
      - Hindsight alignment: 0.03 weight
      - Token efficiency: 0.05 weight
    """
    score = 0.0

    # Enrollment rate (15%)
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.15 * enrollment_pct

    # Retention (10%)
    funnel = final_obs.current_funnel
    total_enrolled_ever = funnel.get("enrolled", 0) + funnel.get("dropped", 0)
    dropped = funnel.get("dropped", 0)
    if total_enrolled_ever > 0:
        retention = 1.0 - (dropped / total_enrolled_ever)
        score += 0.10 * max(0.0, retention)
    else:
        score += 0.0

    # Budget efficiency (10%) - only if agent engaged (screened >= 20)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / _initial_budget(final_obs)
    if screened >= 20:
        if budget_frac >= 0.10:
            score += 0.10
        elif budget_frac >= 0.0:
            score += 0.10 * (budget_frac / 0.10)

    # Dropout recovery (10%) - after a dropout event, did enrollment increase in next 10 steps?
    dropout_steps = [i for i, h in enumerate(history) if h.get("dropout")]
    if dropout_steps:
        recovered = 0
        for di in dropout_steps:
            enrolled_at_drop = history[di].get("enrolled", 0)
            for j in range(di + 1, min(di + 11, len(history))):
                if history[j].get("enrolled", 0) > enrolled_at_drop:
                    recovered += 1
                    break
        score += 0.10 * (recovered / len(dropout_steps))
    # else: no dropouts = no credit (agent must face and survive adversity)

    # Curriculum response (10%) - did agent exploit easy-pool resets?
    curriculum_steps = [i for i, h in enumerate(history) if h.get("curriculum")]
    if curriculum_steps:
        curriculum_screens = sum(
            1
            for ci in curriculum_steps
            if ci < len(history)
            and history[ci].get("action") == "screen_patient"
            and history[ci].get("screen_success")
        )
        max_possible = len(curriculum_steps)
        if max_possible > 0:
            score += 0.10 * min(1.0, curriculum_screens / (max_possible * 0.5))
    else:
        pass  # no curriculum = no credit (hard bench always has curriculum, but no free points)

    # Strategy adaptation (10%) - did agent use adjust_strategy?
    strategy_changes = sum(1 for h in history if h.get("action") == "adjust_strategy")
    if strategy_changes >= 3:
        score += 0.10
    elif strategy_changes >= 1:
        score += 0.10 * (strategy_changes / 3.0)

    # Hypothesis consistency (10%) - especially important in hard (chaotic) bench
    score += 0.10 * _hypothesis_consistency_score(history)

    # Hypothesis accuracy (10%) - did agent identify dropout as dominant?
    score += 0.10 * _hypothesis_accuracy_score(history, final_obs.world_type)

    # Memory + hindsight recovery signals (10%)
    score += 0.04 * _memory_use_score(history)
    score += 0.03 * _milestone_potential_score(final_obs, history)
    score += 0.03 * _hindsight_alignment_score(final_obs, history)

    # Token efficiency under pressure (5%)
    score += 0.05 * _token_efficiency_grade(final_obs, history)

    return round(min(0.999, max(0.001, score)), 4)


GRADERS = {
    "easy_bench": grade_easy_bench,
    "medium_bench": grade_medium_bench,
    "hard_bench": grade_hard_bench,
}
