"""Live HF Space smoke test for the current public API surface."""

import httpx
import sys

BASE = "https://pratimassaravanan-clinical-recruitment.hf.space"
c = httpx.Client(timeout=30, follow_redirects=True)
checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("LIVE SMOKE TEST: Clinical Recruitment (HF Space)")
print("=" * 60)

try:
    # Root
    r = c.get(f"{BASE}/")
    check("Root returns 200", r.status_code == 200)
    data = r.json()
    check("Name correct", data.get("name") == "adaptive-clinical-recruitment")
    check("Root lists exactly 3 tasks", len(data.get("tasks", [])) == 3)

    # Reset
    r = c.post(f"{BASE}/reset", params={"task_id": "easy_bench"})
    check("Reset returns 200", r.status_code == 200)
    d = r.json()
    check("Has observation", "observation" in d)
    check("done=False", d.get("done") is False)

    # Step
    r = c.post(
        f"{BASE}/step",
        json={
            "action_type": "adjust_strategy",
            "strategy_change": "increase_outreach",
        },
    )
    check("Step returns 200", r.status_code == 200)
    d = r.json()
    check("Step has reward", "reward" in d)
    check("Step has causal_insight", "causal_insight" in d.get("observation", {}))
    check("Step has milestones", "milestones" in d.get("observation", {}))
    check(
        "Step has active_constraints",
        "active_constraints" in d.get("observation", {}),
    )
    check(
        "Step has delayed_effects_pending",
        "delayed_effects_pending" in d.get("observation", {}),
    )
    check(
        "Step has patient_memory_summary",
        "patient_memory_summary" in d.get("observation", {}),
    )
    check("Step has current_plan", "current_plan" in d.get("observation", {}))
    check(
        "Step has indexed_memory_summary",
        "indexed_memory_summary" in d.get("observation", {}),
    )
    check(
        "Step has milestone_potential",
        "milestone_potential" in d.get("observation", {}),
    )
    check(
        "Step has token_budget_remaining",
        "token_budget_remaining" in d.get("observation", {}),
    )
    check(
        "Step has token_efficiency_score",
        "token_efficiency_score" in d.get("observation", {}),
    )

    # Tasks
    r = c.get(f"{BASE}/tasks")
    check("Tasks returns 200", r.status_code == 200)
    tasks = r.json()
    check("Tasks returns exactly 3 entries", len(tasks) == 3)
    check(
        "Task ids match current public surface",
        sorted(tasks.keys()) == ["easy_bench", "hard_bench", "medium_bench"],
    )

    # State
    r = c.get(f"{BASE}/state")
    check("State returns 200", r.status_code == 200)
    state = r.json()
    check("State has milestones", "milestones" in state)
    check("State has active_constraints", "active_constraints" in state)
    check("State has current_plan", "current_plan" in state)
    check("State has indexed_memory_summary", "indexed_memory_summary" in state)
    check("State has token_usage_so_far", "token_usage_so_far" in state)

except httpx.ConnectError:
    print(f"\n  [FAIL] Cannot connect to {BASE}")
    print("         The HF Space may not be deployed yet, or the local Python/httpx TLS stack may be failing the handshake.")
    sys.exit(1)
except Exception as e:
    print(f"\n  [FAIL] Error: {e}")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
print("=" * 60)

c.close()
