"""Direct environment logic tests for Adaptive Clinical Recruitment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ClinicalRecruitmentEnv
from models import Action

checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("ENVIRONMENT LOGIC TESTS: Clinical Recruitment")
print("=" * 60)

# 1. Reset all tasks
print("\n1. Reset all tasks")
env = ClinicalRecruitmentEnv()
for task in ["easy_bench", "medium_bench", "hard_bench"]:
    result = env.reset(task=task)
    check(f"{task} resets OK", not result.done)
    check(f"{task} returns observation", result.observation is not None)
    check(f"{task} step=0", result.observation.timestamp == 0)

# 2. Step with valid actions
print("\n2. Step with valid actions")
env.reset("easy_bench")
# Must provide patient_id explicitly now
action = Action(
    action_type="screen_patient",
    patient_id="P-1000",
    hypothesis="noise_dominant",
    confidence=0.7,
)
result = env.step(action)
check("step() returns reward", isinstance(result.reward, float))
check("step() returns done", isinstance(result.done, bool))
check("step() returns info", isinstance(result.info, dict))
check("observation has causal_insight", len(result.observation.causal_insight) > 0)
check(
    "observation has hypothesis_accuracy",
    0.0 <= result.observation.hypothesis_accuracy <= 1.0,
)
check("observation has milestones", isinstance(result.observation.milestones, dict))
check(
    "observation has active_constraints",
    isinstance(result.observation.active_constraints, dict),
)
check(
    "observation has uncertainty_components",
    isinstance(result.observation.uncertainty_components, dict),
)
check(
    "observation has patient_memory_summary",
    isinstance(result.observation.patient_memory_summary, dict),
)
check(
    "observation has counterfactual_hint",
    isinstance(result.observation.counterfactual_hint, str)
    and len(result.observation.counterfactual_hint) > 0,
)

# 3. Grader scores in (0, 1)
print("\n3. Grader scores in (0, 1)")
for task in ["easy_bench", "medium_bench", "hard_bench"]:
    env.reset(task=task)
    for i in range(180):
        if env._done:
            break
        action = Action(
            action_type="screen_patient",
            patient_id=f"P-{1000 + i}",
            hypothesis="noise_dominant",
            confidence=0.6,
        )
        result = env.step(action)
    final_score = result.info.get("final_score")
    check(
        f"{task} grader returns float",
        isinstance(final_score, float),
        f"score={final_score}",
    )
    check(f"{task} score in (0,1)", 0.0 < final_score < 1.0, f"score={final_score}")

# 4. Determinism check
print("\n4. Determinism (two identical runs)")
rewards_a = []
env.reset("medium_bench")
for i in range(20):
    if env._done:
        break
    action = Action(
        action_type="screen_patient",
        patient_id=f"P-{1000 + i}",
        hypothesis="site_bias",
        confidence=0.7,
    )
    result = env.step(action)
    rewards_a.append(result.reward)

rewards_b = []
env.reset("medium_bench")
for i in range(20):
    if env._done:
        break
    action = Action(
        action_type="screen_patient",
        patient_id=f"P-{1000 + i}",
        hypothesis="site_bias",
        confidence=0.7,
    )
    result = env.step(action)
    rewards_b.append(result.reward)

check(
    "Deterministic rewards",
    rewards_a == rewards_b,
    f"a={rewards_a[:5]} b={rewards_b[:5]}",
)

# 5. Hypothesis tracking works
print("\n5. Hypothesis tracking")
env.reset("easy_bench")
for idx, hyp in enumerate(
    [
        "noise_dominant",
        "noise_dominant",
        "dropout_dominant",
        "site_bias",
        "noise_dominant",
    ]
):
    action = Action(
        action_type="screen_patient",
        patient_id=f"P-{1000 + idx}",
        hypothesis=hyp,
        confidence=0.6,
    )
    env.step(action)
check("Hypothesis history tracked", len(env._hypothesis_history) == 5)
check(
    "Consistency penalty applied",
    env._consistency_penalty() > 0,
    f"penalty={env._consistency_penalty()}",
)

# 6. Delayed effects and strategy negotiation
print("\n6. Delayed effects and site negotiation")
env.reset("medium_bench")
before_conv = env._sites["site_A"]["conversion_rate"]
result = env.step(
    Action(
        action_type="adjust_strategy",
        strategy_change="negotiate_site_A",
        hypothesis="site_bias",
        confidence=0.8,
    )
)
check(
    "negotiation schedules delayed effect",
    result.observation.delayed_effects_pending >= 1,
    f"pending={result.observation.delayed_effects_pending}",
)
env.step(
    Action(
        action_type="screen_patient",
        patient_id="P-1000",
        hypothesis="site_bias",
        confidence=0.7,
    )
)
result = env.step(
    Action(
        action_type="screen_patient",
        patient_id="P-1001",
        hypothesis="site_bias",
        confidence=0.7,
    )
)
after_conv = env._sites["site_A"]["conversion_rate"]
check(
    "site negotiation improves conversion after delay",
    after_conv > before_conv,
    f"before={before_conv} after={after_conv}",
)
check(
    "delayed effect surfaced in info",
    result.info.get("reward_breakdown", {}).get("delayed_effects_triggered", 0) >= 1,
)

# 7. State endpoint
print("\n7. State check")
state = env.state()
check("state() returns task", state.task == "medium_bench")
check("state() returns step", state.step == 3)
check("state() returns history", len(state.history) > 0)
check("state() exposes milestones", isinstance(state.milestones, dict))
check(
    "state() exposes active_constraints",
    isinstance(state.active_constraints, dict),
)

# 8. World type per task
print("\n8. World type per task")
for task, expected in [
    ("easy_bench", "noise"),
    ("medium_bench", "site_bias"),
    ("hard_bench", "dropout"),
]:
    env.reset(task)
    check(f"{task} world_type={expected}", env._world_type == expected)

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    print("ALL CHECKS PASSED!")
else:
    print("SOME CHECKS FAILED - review above")
print("=" * 60)
