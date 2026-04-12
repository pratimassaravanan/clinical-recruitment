"""Live HF Spaces smoke test for Clinical Recruitment."""

import httpx
import sys

BASE = "https://kaushikss-clinical-recruitment.hf.space"
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
            "action_type": "screen_patient",
            "hypothesis": "noise_dominant",
            "confidence": 0.7,
        },
    )
    check("Step returns 200", r.status_code == 200)
    d = r.json()
    check("Step has reward", "reward" in d)
    check("Step has causal_insight", "causal_insight" in d.get("observation", {}))

    # Tasks
    r = c.get(f"{BASE}/tasks")
    check("Tasks returns 200", r.status_code == 200)
    check("3+ tasks", len(r.json()) >= 3)

    # State
    r = c.get(f"{BASE}/state")
    check("State returns 200", r.status_code == 200)

except httpx.ConnectError:
    print(f"\n  [FAIL] Cannot connect to {BASE}")
    print("         The HF Space may not be deployed yet.")
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
