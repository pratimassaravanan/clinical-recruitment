"""Deterministic graders for Clinical Recruitment tasks. Each returns scores in (0, 1) with partial credit.

Grading includes hypothesis accuracy and reasoning consistency (Upgrades 1-4).
"""

from models import Observation


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


def grade_easy_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade easy bench: stable patient pool, generous budget/time.

    Scoring (0.0-1.0):
      - Enrollment rate: 0.35 weight (% of target reached)
      - Budget efficiency: 0.20 weight (budget remaining / initial)
      - Screening accuracy: 0.20 weight (eligible / screened ratio)
      - Timeline: 0.10 weight (enrolled before deadline)
      - Hypothesis consistency: 0.10 weight (stable reasoning)
      - Hypothesis accuracy: 0.05 weight (correct world model)
    """
    score = 0.0

    # Enrollment rate (35%) - how much of target was reached
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.35 * enrollment_pct

    # Budget efficiency (20%) - reward for not exhausting budget
    # BUT: only award if agent actually took productive actions (screened >= 10 patients)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / 120000.0
    if screened >= 10:
        if budget_frac >= 0.20:
            score += 0.20
        elif budget_frac >= 0.0:
            score += 0.20 * (budget_frac / 0.20)
    # else: 0 — no free points for doing nothing

    # Screening accuracy (20%) - eligible / screened ratio
    eligible = funnel.get("eligible", 0)
    if screened > 0:
        accuracy = eligible / screened
        score += 0.20 * min(1.0, accuracy / 0.7)
    else:
        score += 0.0

    # Timeline (10%) - bonus for finishing early
    days_used = len(history)
    if enrolled >= target:
        time_frac = 1.0 - (days_used / 180.0)
        score += 0.10 * time_frac
    elif days_used < 180:
        score += 0.10 * (enrolled / max(1, target)) * 0.5

    # Hypothesis consistency (10%)
    score += 0.10 * _hypothesis_consistency_score(history)

    # Hypothesis accuracy (5%)
    score += 0.05 * _hypothesis_accuracy_score(history, final_obs.world_type)

    return round(min(0.999, max(0.001, score)), 4)


def grade_medium_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade medium bench: multiple sites, moderate dropout, uncertainty.

    Scoring (0.0-1.0):
      - Enrollment rate: 0.30 weight
      - Retention: 0.20 weight (enrolled - dropped / enrolled)
      - Site utilization: 0.15 weight (used multiple sites effectively)
      - Budget efficiency: 0.15 weight
      - Hypothesis consistency: 0.10 weight
      - Hypothesis accuracy: 0.10 weight
    """
    score = 0.0

    # Enrollment rate (30%)
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.30 * enrollment_pct

    # Retention (20%) - low dropout
    funnel = final_obs.current_funnel
    total_enrolled_ever = funnel.get("enrolled", 0) + funnel.get("dropped", 0)
    dropped = funnel.get("dropped", 0)
    if total_enrolled_ever > 0:
        retention = 1.0 - (dropped / total_enrolled_ever)
        score += 0.20 * max(0.0, retention)
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

    # Budget efficiency (15%) - only if agent engaged (screened >= 15)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / 150000.0
    if screened >= 15:
        if budget_frac >= 0.15:
            score += 0.15
        elif budget_frac >= 0.0:
            score += 0.15 * (budget_frac / 0.15)

    # Hypothesis consistency (10%)
    score += 0.10 * _hypothesis_consistency_score(history)

    # Hypothesis accuracy (10%)
    score += 0.10 * _hypothesis_accuracy_score(history, final_obs.world_type)

    return round(min(0.999, max(0.001, score)), 4)


def grade_hard_bench(
    final_obs: Observation, total_reward: float, history: list
) -> float:
    """Grade hard bench: tight budget, high dropout, curriculum injections.

    Scoring (0.0-1.0):
      - Enrollment rate: 0.20 weight
      - Retention: 0.15 weight
      - Budget efficiency: 0.15 weight
      - Dropout recovery: 0.10 weight (bounced back after high-dropout events)
      - Curriculum response: 0.10 weight (exploited easy-pool resets)
      - Strategy adaptation: 0.10 weight (used strategy changes effectively)
      - Hypothesis consistency: 0.10 weight (stable reasoning under chaos)
      - Hypothesis accuracy: 0.10 weight (identified dropout as dominant)
    """
    score = 0.0

    # Enrollment rate (20%)
    target = final_obs.target_enrollment
    enrolled = final_obs.enrolled_so_far
    if target > 0:
        enrollment_pct = min(1.0, enrolled / target)
        score += 0.20 * enrollment_pct

    # Retention (15%)
    funnel = final_obs.current_funnel
    total_enrolled_ever = funnel.get("enrolled", 0) + funnel.get("dropped", 0)
    dropped = funnel.get("dropped", 0)
    if total_enrolled_ever > 0:
        retention = 1.0 - (dropped / total_enrolled_ever)
        score += 0.15 * max(0.0, retention)
    else:
        score += 0.0

    # Budget efficiency (15%) - only if agent engaged (screened >= 20)
    funnel = final_obs.current_funnel
    screened = funnel.get("screened", 0)
    budget_frac = max(0, final_obs.budget_remaining) / 100000.0
    if screened >= 20:
        if budget_frac >= 0.10:
            score += 0.15
        elif budget_frac >= 0.0:
            score += 0.15 * (budget_frac / 0.10)

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

    return round(min(0.999, max(0.001, score)), 4)


GRADERS = {
    "easy_bench": grade_easy_bench,
    "medium_bench": grade_medium_bench,
    "hard_bench": grade_hard_bench,
}
