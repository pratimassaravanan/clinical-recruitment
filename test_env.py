"""Direct environment logic tests for Adaptive Clinical Recruitment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ClinicalRecruitmentEnv
from models import Action
from load_traces import (
    build_progressive_trace,
    get_stage_horizon_days,
    list_progressive_stage_tasks,
    make_stage_task_id,
)
from training.offline_policy import LinearPolicy, MLPPolicy, extract_features
from training.train_offline_policy import train_policy
from training.trajectory_splitter import split_trajectory

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
check(
    "observation has current_plan",
    isinstance(result.observation.current_plan, dict),
)
check(
    "observation has indexed_memory_summary",
    isinstance(result.observation.indexed_memory_summary, dict),
)
check(
    "observation has retrieved_memory_context",
    isinstance(result.observation.retrieved_memory_context, str),
)
check(
    "observation has milestone_potential",
    0.0 <= result.observation.milestone_potential <= 1.0,
)
check(
    "observation has active_milestone",
    isinstance(result.observation.active_milestone, str),
)
check(
    "observation has hindsight_available",
    isinstance(result.observation.hindsight_available, bool),
)
check(
    "observation has token_budget_remaining",
    isinstance(result.observation.token_budget_remaining, int),
)
check(
    "observation has token_usage_so_far",
    isinstance(result.observation.token_usage_so_far, int),
)
check(
    "observation has token_efficiency_score",
    0.0 <= result.observation.token_efficiency_score <= 1.0,
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
check("state() exposes current_plan", isinstance(state.current_plan, dict))
check(
    "state() exposes indexed_memory_summary",
    isinstance(state.indexed_memory_summary, dict),
)
check(
    "state() exposes milestone_potential",
    0.0 <= state.milestone_potential <= 1.0,
)
check("state() exposes token_usage", state.token_usage_so_far >= 0)
check(
    "state() exposes token_efficiency_score",
    0.0 <= state.token_efficiency_score <= 1.0,
)

# 8. Explicit plan and memory actions
print("\n8. Plan and memory actions")
env.reset("medium_bench")
result = env.step(
    Action(
        action_type="plan_next_phase",
        target_phase="conversion",
        plan_id="manual-convert",
        plan_summary="Convert pending eligible patients before they cool off.",
        hypothesis="site_bias",
        confidence=0.7,
    )
)
check(
    "plan action updates observation current_plan",
    result.observation.current_plan.get("target_phase") == "conversion",
)
check("plan action has bounded reward", -0.2 <= result.reward <= 0.5, f"reward={result.reward}")
result = env.step(
    Action(
        action_type="summarize_and_index",
        memory_key="conversion_note",
        memory_payload="followup_due patients need conversion focus",
        hypothesis="site_bias",
        confidence=0.6,
    )
)
check(
    "memory summarize creates indexed entry",
    result.observation.indexed_memory_summary.get("entries", 0) >= 1,
)
check(
    "memory summarize has bounded reward",
    -0.2 <= result.reward <= 0.5,
    f"reward={result.reward}",
)
result = env.step(
    Action(
        action_type="retrieve_relevant_history",
        memory_query="conversion followup",
        hypothesis="site_bias",
        confidence=0.6,
    )
)
check(
    "memory retrieval exposes context",
    len(result.observation.retrieved_memory_context) > 0,
    result.observation.retrieved_memory_context,
)
check(
    "token usage increases after cognitive actions",
    result.observation.token_usage_so_far > 0,
    f"tokens={result.observation.token_usage_so_far}",
)

# 9. Hindsight summary on terminal step
print("\n9. Hindsight summary")
env.reset("easy_bench")
result = env.step(
    Action(
        action_type="stop_recruitment",
        hypothesis="noise_dominant",
        confidence=0.2,
    )
)
check("stop_recruitment ends episode", result.done)
check(
    "terminal info includes hindsight_summary",
    isinstance(result.info.get("hindsight_summary"), dict),
)
check(
    "terminal observation exposes hindsight_available",
    result.observation.hindsight_available is True,
)

# 10. World type per task
print("\n10. World type per task")
for task, expected in [
    ("easy_bench", "noise"),
    ("medium_bench", "site_bias"),
    ("hard_bench", "dropout"),
]:
    env.reset(task)
    check(f"{task} world_type={expected}", env._world_type == expected)

# 11. Progressive horizon tasks and trajectory splitting
print("\n11. Progressive staging")
stage_task = make_stage_task_id("medium_bench", 90)
trace = build_progressive_trace("medium_bench", 90)
check("stage task id encodes horizon", stage_task == "medium_bench_stage_90")
check("stage horizon parsed", get_stage_horizon_days(stage_task) == 90)
check("progressive trace horizon clipped", trace["deadline_days"] == 90)
check(
    "progressive task list includes hard stage",
    "hard_bench_stage_180" in list_progressive_stage_tasks(),
)
result = env.reset(stage_task)
check("stage task resets OK", not result.done)
check("stage task has clipped deadline", env._max_steps == 90, f"max_steps={env._max_steps}")
for idx in range(6):
    env.step(
        Action(
            action_type="screen_patient",
            patient_id=f"P-{1000 + idx}",
            hypothesis="site_bias",
            confidence=0.6,
        )
    )
chunks = split_trajectory(env.get_history(), window=4, overlap=2)
check("trajectory splitter yields chunks", len(chunks) >= 2, f"chunks={len(chunks)}")
check(
    "trajectory chunk carries boundary context",
    isinstance(chunks[0].get("boundary_context"), dict),
)

# 12. Token-aware training helpers
print("\n12. Token-aware training helpers")
env.reset("medium_bench")
features = extract_features(env.state().model_dump(), 0)
check("feature extractor includes token_budget_scaled", "token_budget_scaled" in features)
check("feature extractor includes token_efficiency", "token_efficiency" in features)
policy = LinearPolicy(seed=1)
chosen = policy.choose_action_type(features, epsilon=0.0)
check("linear policy chooses known action", isinstance(chosen, str) and len(chosen) > 0)
mlp_policy = MLPPolicy(seed=1)
mlp_choice = mlp_policy.choose_action_type(features, epsilon=0.0)
check("mlp policy chooses known action", isinstance(mlp_choice, str) and len(mlp_choice) > 0)
trained_policy, history = train_policy(["medium_bench_stage_30"], epochs=2, seed=1)
check("offline trainer returns history", len(history) == 2)
check("offline trainer returns policy weights", isinstance(trained_policy.weights, dict))

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
