"""Local serving test - runs full inference loop without external deployment.

Tests the environment and heuristic policy locally to validate the serving path.
"""

import sys
import time
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ClinicalRecruitmentEnv
from models import Action, Observation


def heuristic_policy(obs: Observation) -> dict:
    """Simple heuristic policy for testing the inference loop."""
    # Prioritize based on funnel state
    funnel = obs.current_funnel
    patients = obs.available_patients
    recontact_candidates = obs.recontact_candidates
    allocation_candidates = obs.allocation_candidates
    sites = obs.site_performance
    
    # Decision logic
    screened = funnel.get("screened", 0)
    eligible = funnel.get("eligible", 0)
    consented = funnel.get("consented", 0)

    # Have consented patients, try to allocate
    if allocation_candidates and sites:
        patient = allocation_candidates[0]
        patient_id = patient.get("id") if isinstance(patient, dict) else patient.id
        site_id = list(sites.keys())[0]
        return {
            "action_type": "allocate_to_site",
            "patient_id": patient_id,
            "site_id": site_id,
        }
    
    # Recontact for conversion
    if recontact_candidates and random.random() < 0.3:
        patient = recontact_candidates[0]
        patient_id = patient.get("id") if isinstance(patient, dict) else patient.id
        return {"action_type": "recontact", "patient_id": patient_id}

    # If no screenable patients remain, adjust strategy to surface more candidates.
    if not patients:
        return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach"}

    # Need more screening
    if screened < 10 or eligible <= consented:
        patient = patients[0]
        patient_id = patient.get("id") if isinstance(patient, dict) else patient.id
        return {
            "action_type": "screen_patient",
            "patient_id": patient_id,
            "hypothesis": random.choice(["noise_dominant", "site_bias", "dropout"]),
            "confidence": 0.7,
        }
    
    # Default: screen more
    patient = patients[0]
    patient_id = patient.get("id") if isinstance(patient, dict) else patient.id
    return {
        "action_type": "screen_patient",
        "patient_id": patient_id,
        "hypothesis": "noise_dominant",
        "confidence": 0.6,
    }

checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 70)
print("LOCAL SERVING TEST: Clinical Recruitment")
print("=" * 70)

env = ClinicalRecruitmentEnv()

# Test all task difficulties
for task_id in ["easy_bench", "medium_bench", "hard_bench"]:
    print(f"\n{task_id.upper()}")
    print("-" * 40)
    
    result = env.reset(task=task_id)
    obs = result.observation
    
    # Check observation has all long-horizon fields
    check(f"{task_id}: has milestones", obs.milestones is not None)
    check(f"{task_id}: has active_constraints", obs.active_constraints is not None)
    check(f"{task_id}: has delayed_effects_pending", obs.delayed_effects_pending is not None)
    check(f"{task_id}: has patient_memory_summary", obs.patient_memory_summary is not None)
    check(f"{task_id}: has current_plan", obs.current_plan is not None)
    check(f"{task_id}: has indexed_memory_summary", obs.indexed_memory_summary is not None)
    check(f"{task_id}: has milestone_potential", obs.milestone_potential is not None)
    check(f"{task_id}: has token_budget_remaining", obs.token_budget_remaining is not None)
    check(f"{task_id}: has token_efficiency_score", obs.token_efficiency_score is not None)
    check(f"{task_id}: has causal_insight", obs.causal_insight is not None)
    check(f"{task_id}: has uncertainty_components", obs.uncertainty_components is not None)
    check(f"{task_id}: has counterfactual_hint", obs.counterfactual_hint is not None)

print("\n" + "=" * 70)
print("HEURISTIC POLICY INTEGRATION TEST")
print("=" * 70)

# Run inference loop for multiple steps
result = env.reset(task="medium_bench")
obs = result.observation
done = result.done

step_count = 0
action_types_used = set()
total_reward = 0.0

print("\nRunning policy loop...")
while not done and step_count < 30:
    # Call heuristic policy
    action_dict = heuristic_policy(obs)
    
    # Validate action
    check(f"step_{step_count}: valid action_type", "action_type" in action_dict)
    
    # Build action and step
    action = Action(**action_dict)
    result = env.step(action)
    obs = result.observation
    done = result.done
    
    action_types_used.add(action_dict.get("action_type", "unknown"))
    total_reward += result.reward
    step_count += 1
    
    # Print progress every 5 steps
    if step_count % 5 == 0:
        print(f"  Step {step_count}: reward={result.reward:.4f}, enrolled={obs.enrolled_so_far}")

print(f"\n  Total steps: {step_count}")
print(f"  Total reward: {total_reward:.4f}")
print(f"  Action types used: {action_types_used}")
print(f"  Final enrolled: {obs.enrolled_so_far}")

check("policy: ran multiple steps", step_count >= 10)
check("policy: used multiple action types", len(action_types_used) >= 2)
check("policy: non-zero total reward", total_reward != 0)
check("policy: enrolled patients", obs.enrolled_so_far > 0)

print("\n" + "=" * 70)
print("ENVIRONMENT STATE PERSISTENCE TEST")
print("=" * 70)

# Test that environment state persists correctly
env2 = ClinicalRecruitmentEnv()
result = env2.reset(task="easy_bench")
obs1 = result.observation

# First action
action1 = heuristic_policy(obs1)
result1 = env2.step(Action(**action1))

# Check state updated
check("state: step incremented", result1.observation.timestamp > 0)
check("state: budget changed", result1.observation.budget_remaining <= obs1.budget_remaining)

# Second action
action2 = heuristic_policy(result1.observation)
result2 = env2.step(Action(**action2))

check("state: consecutive steps work", result2 is not None)
check("state: action has valid type", action2.get("action_type") in [
    "screen_patient", "recontact", "allocate_to_site", "adjust_strategy",
    "stop_recruitment", "plan_next_phase", "summarize_and_index",
    "retrieve_relevant_history"
])

print("\n" + "=" * 70)
print("FULL EPISODE TEST")
print("=" * 70)

# Run complete episode
env3 = ClinicalRecruitmentEnv()
result = env3.reset(task="easy_bench")
obs = result.observation
done = result.done
episode_reward = 0.0
episode_steps = 0

start_time = time.time()
while not done and episode_steps < 200:
    action_dict = heuristic_policy(obs)
    action = Action(**action_dict)
    result = env3.step(action)
    obs = result.observation
    done = result.done
    episode_reward += result.reward
    episode_steps += 1

elapsed = time.time() - start_time
final_score = result.info.get("final_score", 0.0)

print(f"  Episode completed in {episode_steps} steps ({elapsed:.2f}s)")
print(f"  Final score: {final_score:.4f}")
print(f"  Total reward: {episode_reward:.4f}")
print(f"  Enrolled: {obs.enrolled_so_far}/{obs.target_enrollment}")

check("episode: completed without errors", True)
check("episode: has final score", final_score > 0)
check("episode: reasonable steps", 10 < episode_steps < 200)

# Summary
print("\n" + "=" * 70)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    print("ALL CHECKS PASSED!")
print("=" * 70)
