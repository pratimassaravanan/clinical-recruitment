"""Local HTTP API endpoint integration tests for Clinical Recruitment."""

import httpx
import sys

BASE = "http://localhost:7860"
c = httpx.Client(timeout=30)
checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("LOCAL API TESTS: Clinical Recruitment (localhost:7860)")
print("=" * 60)

try:
    # 1. Root
    print("\n1. Root endpoint")
    r = c.get(f"{BASE}/")
    check("GET / returns 200", r.status_code == 200)
    data = r.json()
    check("name is correct", data.get("name") == "adaptive-clinical-recruitment")
    check("tasks listed", len(data.get("tasks", [])) == 3)

    # 2. Health
    print("\n2. Health endpoint")
    r = c.get(f"{BASE}/health")
    check("GET /health returns 200", r.status_code == 200)

    # 3. Tasks
    print("\n3. Tasks endpoint")
    r = c.get(f"{BASE}/tasks")
    tasks = r.json()
    check("3 tasks defined", len(tasks) >= 3)

    # 4. Reset
    print("\n4. Reset endpoint")
    for task_id in ["easy_bench", "medium_bench", "hard_bench"]:
        r = c.post(f"{BASE}/reset", params={"task_id": task_id})
        check(f"reset({task_id}) returns 200", r.status_code == 200)
        d = r.json()
        check(f"reset({task_id}) has observation", "observation" in d)

    # 5. Step
    print("\n5. Step endpoint")
    r = c.post(f"{BASE}/reset", params={"task_id": "easy_bench"})
    r = c.post(
        f"{BASE}/step",
        json={
            "action_type": "screen_patient",
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
            "hypothesis": "noise_dominant",
            "confidence": 0.7,
        },
    )
    d = r.json()
    check("step() returns 200", r.status_code == 200)
    check("step() has observation", "observation" in d)
    check("step() has reward", "reward" in d)
    check("step() has done", "done" in d)
    check("step() has info", "info" in d)
    check(
        "observation has causal_insight", "causal_insight" in d.get("observation", {})
    )
    check(
        "observation has hypothesis_accuracy",
        "hypothesis_accuracy" in d.get("observation", {}),
    )
    check("observation has milestones", "milestones" in d.get("observation", {}))
    check(
        "observation has active_constraints",
        "active_constraints" in d.get("observation", {}),
    )
    check(
        "observation has delayed_effects_pending",
        "delayed_effects_pending" in d.get("observation", {}),
    )
    check(
        "observation has uncertainty_components",
        "uncertainty_components" in d.get("observation", {}),
    )
    check(
        "observation has patient_memory_summary",
        "patient_memory_summary" in d.get("observation", {}),
    )
    check(
        "observation has counterfactual_hint",
        "counterfactual_hint" in d.get("observation", {}),
    )
    check("observation has current_plan", "current_plan" in d.get("observation", {}))
    check(
        "observation has indexed_memory_summary",
        "indexed_memory_summary" in d.get("observation", {}),
    )
    check(
        "observation has retrieved_memory_context",
        "retrieved_memory_context" in d.get("observation", {}),
    )
    check(
        "observation has milestone_potential",
        "milestone_potential" in d.get("observation", {}),
    )
    check(
        "observation has active_milestone",
        "active_milestone" in d.get("observation", {}),
    )
    check(
        "observation has hindsight_available",
        "hindsight_available" in d.get("observation", {}),
    )
    check(
        "observation has token_budget_remaining",
        "token_budget_remaining" in d.get("observation", {}),
    )
    check(
        "observation has token_usage_so_far",
        "token_usage_so_far" in d.get("observation", {}),
    )
    check(
        "observation has token_efficiency_score",
        "token_efficiency_score" in d.get("observation", {}),
    )

    # 5b. Plan and memory actions
    print("\n5b. Plan/memory actions")
    r = c.post(
        f"{BASE}/step",
        json={
            "action_type": "plan_next_phase",
            "target_phase": "conversion",
            "plan_id": "api-plan",
            "plan_summary": "Convert warm leads first",
            "hypothesis": "noise_dominant",
            "confidence": 0.6,
        },
    )
    d = r.json()
    check("plan_next_phase returns 200", r.status_code == 200)
    check(
        "plan_next_phase updates current_plan",
        d.get("observation", {}).get("current_plan", {}).get("target_phase") == "conversion",
    )

    r = c.post(
        f"{BASE}/step",
        json={
            "action_type": "summarize_and_index",
            "memory_key": "api_memory",
            "memory_payload": "conversion followup backlog",
            "hypothesis": "noise_dominant",
            "confidence": 0.6,
        },
    )
    d = r.json()
    check("summarize_and_index returns 200", r.status_code == 200)
    check(
        "summarize_and_index updates memory summary",
        d.get("observation", {}).get("indexed_memory_summary", {}).get("entries", 0) >= 1,
    )

    r = c.post(
        f"{BASE}/step",
        json={
            "action_type": "retrieve_relevant_history",
            "memory_query": "conversion followup",
            "hypothesis": "noise_dominant",
            "confidence": 0.6,
        },
    )
    d = r.json()
    check("retrieve_relevant_history returns 200", r.status_code == 200)
    check(
        "retrieve_relevant_history exposes context",
        len(d.get("observation", {}).get("retrieved_memory_context", "")) > 0,
    )

    # 6. State
    print("\n6. State endpoint")
    r = c.get(f"{BASE}/state")
    d = r.json()
    check("state() returns 200", r.status_code == 200)
    check("state() has task", "task" in d)
    check("state() has step", "step" in d)
    check("state() has milestones", "milestones" in d)
    check("state() has active_constraints", "active_constraints" in d)
    check("state() has delayed_effects_pending", "delayed_effects_pending" in d)
    check("state() has uncertainty_components", "uncertainty_components" in d)
    check("state() has current_plan", "current_plan" in d)
    check("state() has indexed_memory_summary", "indexed_memory_summary" in d)
    check("state() has milestone_potential", "milestone_potential" in d)
    check("state() has token_usage_so_far", "token_usage_so_far" in d)
    check("state() has token_efficiency_score", "token_efficiency_score" in d)

except httpx.ConnectError:
    print("\n  [FAIL] Cannot connect to localhost:7860. Start the server first:")
    print("         uvicorn app:app --host 0.0.0.0 --port 7860")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
print("=" * 60)

c.close()
