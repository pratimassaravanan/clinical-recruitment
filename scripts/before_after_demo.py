"""Before/After demo: random baseline vs heuristic policy across all 3 tasks.

Shows that the environment's graders differentiate between good and bad
policies -- evidence that training improves agent behaviour.
"""
import sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ClinicalRecruitmentEnv
from models import Action, Observation

random.seed(42)
TASKS = ["easy_bench", "medium_bench", "hard_bench"]
ACTION_TYPES = ["screen_patient", "recontact", "allocate_to_site", "adjust_strategy"]
STRATEGIES = ["increase_outreach", "relax_criteria", "tighten_criteria"]


# ── random agent ─────────────────────────────────────────────────────
def random_policy(obs: Observation) -> dict:
    """Pick a uniformly random legal action each step."""
    pts = obs.available_patients
    rc = obs.recontact_candidates
    ac = obs.allocation_candidates
    sites = list(obs.site_performance.keys())
    c = random.choice(ACTION_TYPES)
    if c == "allocate_to_site" and ac and sites:
        return {"action_type": "allocate_to_site",
                "patient_id": random.choice(ac)["id"],
                "site_id": random.choice(sites)}
    if c == "recontact" and rc:
        return {"action_type": "recontact", "patient_id": random.choice(rc)["id"]}
    if c == "screen_patient" and pts:
        return {"action_type": "screen_patient", "patient_id": random.choice(pts)["id"]}
    if c == "adjust_strategy":
        return {"action_type": "adjust_strategy",
                "strategy_change": random.choice(STRATEGIES)}
    if pts:
        return {"action_type": "screen_patient", "patient_id": random.choice(pts)["id"]}
    return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach"}


# ── heuristic agent ──────────────────────────────────────────────────
_site_rr = 0
_step_ctr = 0
_alloc_tries: dict = {}


def heuristic_policy(obs: Observation) -> dict:
    """Consistent hypothesis, site diversity, screen-first funnel,
    periodic strategy changes, capped allocation retries."""
    global _site_rr, _step_ctr, _alloc_tries
    _step_ctr += 1
    pts = obs.available_patients
    rc = obs.recontact_candidates
    ac = obs.allocation_candidates
    sids = sorted(obs.site_performance.keys())
    hyp_map = {"dropout": "dropout_dominant", "noise": "noise_dominant",
               "site_bias": "site_bias"}
    hyp = hyp_map.get(obs.world_type, "noise_dominant")
    base = {"hypothesis": hyp, "confidence": 0.7}

    # Periodic strategy changes (hard_bench grader rewards >= 3)
    if _step_ctr % 40 == 0:
        strat = STRATEGIES[(_step_ctr // 40) % 3]
        return {**base, "action_type": "adjust_strategy", "strategy_change": strat}

    # 1. Allocate consented patients (max 2 attempts each, rotate sites)
    for p in ac:
        pid = p["id"]
        if _alloc_tries.get(pid, 0) < 2 and sids:
            sid = sids[_site_rr % len(sids)]; _site_rr += 1
            _alloc_tries[pid] = _alloc_tries.get(pid, 0) + 1
            return {**base, "action_type": "allocate_to_site",
                    "patient_id": pid, "site_id": sid}

    # 2. Screen new patients -- primary pipeline builder
    if pts:
        return {**base, "action_type": "screen_patient",
                "patient_id": pts[0]["id"]}

    # 3. Recontact consented or follow-up-due patients
    for p in rc:
        if p.get("consented") or p.get("followup_due_day") is not None:
            return {**base, "action_type": "recontact", "patient_id": p["id"]}
    if rc:
        return {**base, "action_type": "recontact", "patient_id": rc[0]["id"]}

    # 4. Fallback
    return {**base, "action_type": "adjust_strategy",
            "strategy_change": "increase_outreach"}


# ── episode runner ───────────────────────────────────────────────────
def run_episode(env, task, policy_fn):
    """Run one full episode and return the grader's final_score."""
    global _site_rr, _step_ctr, _alloc_tries
    _site_rr, _step_ctr, _alloc_tries = 0, 0, {}
    result = env.reset(task=task)
    obs, done = result.observation, result.done
    while not done:
        result = env.step(Action(**policy_fn(obs)))
        obs, done = result.observation, result.done
    return result.info.get("final_score", 0.0)


# ── main ─────────────────────────────────────────────────────────────
def main():
    env = ClinicalRecruitmentEnv()
    rows = []
    for task in TASKS:
        random.seed(42)
        r = run_episode(env, task, random_policy)
        random.seed(42)
        h = run_episode(env, task, heuristic_policy)
        rows.append((task, r, h))

    hdr = f"{'Task':<16} {'Random':>10} {'Heuristic':>10} {'Improv %':>10}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    imps = []
    for task, r, h in rows:
        pct = ((h - r) / r * 100) if r > 0 else (float("inf") if h > 0 else 0.0)
        imps.append(pct)
        print(f"{task:<16} {r:>10.4f} {h:>10.4f} {pct:>+9.1f}%")
    print(sep)
    finite = [p for p in imps if p != float("inf")]
    avg = sum(finite) / len(finite) if finite else 0.0
    print(f"\nHeuristic policy improves over random by "
          f"{avg:+.1f}% average across {len(TASKS)} tasks\n")


if __name__ == "__main__":
    main()
