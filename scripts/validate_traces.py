"""Validate the 5k traces file for structural issues."""
import json
from pathlib import Path

p = Path("data/sft_traces_5k_v2.json")
data = json.loads(p.read_text())
traces = data["traces"]
stats = data.get("stats", {})

print(f"Total traces: {len(traces)}")
print(f"Stats: {json.dumps(stats, indent=2)}")
print()

bad_roles = 0
bad_first = 0
no_assistant = 0
no_user = 0
empty_traces = 0
short_traces = 0
bad_json_actions = 0
good_json_actions = 0
action_types = {}
traces_by_task = {}
total_messages = 0
sample_bad = []

for i, trace in enumerate(traces):
    if not trace:
        empty_traces += 1
        continue
    if len(trace) < 3:
        short_traces += 1

    if trace[0].get("role") != "system":
        bad_first += 1
        if len(sample_bad) < 3:
            sample_bad.append(f"trace {i}: first role={trace[0].get('role')}")

    has_assistant = False
    has_user = False
    for msg in trace:
        total_messages += 1
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role not in ("system", "user", "assistant"):
            bad_roles += 1

        if role == "assistant":
            has_assistant = True
            try:
                parsed = json.loads(content)
                if "action_type" in parsed:
                    good_json_actions += 1
                    at = parsed["action_type"]
                    action_types[at] = action_types.get(at, 0) + 1
                else:
                    bad_json_actions += 1
            except json.JSONDecodeError:
                bad_json_actions += 1
                if len(sample_bad) < 6:
                    sample_bad.append(f"trace {i}: bad JSON: {content[:60]}")

        if role == "user" and "step=" in content:
            has_user = True
            if "budget=120000" in content:
                traces_by_task["easy_bench"] = traces_by_task.get("easy_bench", 0) + 1
            elif "budget=150000" in content:
                traces_by_task["medium_bench"] = traces_by_task.get("medium_bench", 0) + 1
            elif "budget=100000" in content:
                traces_by_task["hard_bench"] = traces_by_task.get("hard_bench", 0) + 1

    if not has_assistant:
        no_assistant += 1
    if not has_user:
        no_user += 1

print("=== STRUCTURE ===")
print(f"Total messages: {total_messages}")
print(f"Empty traces: {empty_traces}")
print(f"Short traces (<3 msgs): {short_traces}")
print(f"Bad first role (not system): {bad_first}")
print(f"Bad roles: {bad_roles}")
print(f"No assistant: {no_assistant}")
print(f"No user obs: {no_user}")
print()
print("=== ACTIONS ===")
print(f"Valid JSON: {good_json_actions}")
print(f"Bad/non-JSON: {bad_json_actions}")
print(json.dumps(action_types, indent=2))
print()
print("=== TASK DISTRIBUTION ===")
print(json.dumps(traces_by_task, indent=2))
print()
if sample_bad:
    print("=== SAMPLE ISSUES ===")
    for s in sample_bad:
        print(f"  {s}")
else:
    print("NO ISSUES FOUND")

# Spot check: verify patient IDs in actions match observation
print()
print("=== ID HALLUCINATION CHECK (sample 100 traces) ===")
hallucinated = 0
checked = 0
import random
random.seed(42)
sample_indices = random.sample(range(len(traces)), min(100, len(traces)))
for idx in sample_indices:
    trace = traces[idx]
    last_obs_ids = set()
    for msg in trace:
        if msg["role"] == "user" and "step=" in msg["content"]:
            # Extract all P-XXXX from user obs
            import re
            last_obs_ids = set(re.findall(r"P-\d+", msg["content"]))
        elif msg["role"] == "assistant":
            try:
                action = json.loads(msg["content"])
                pid = action.get("patient_id")
                if pid and last_obs_ids and pid not in last_obs_ids:
                    hallucinated += 1
                    checked += 1
                elif pid:
                    checked += 1
            except json.JSONDecodeError:
                pass

print(f"Checked {checked} actions with patient_id")
print(f"Hallucinated IDs (not in preceding obs): {hallucinated}")
if checked > 0:
    print(f"Hallucination rate: {hallucinated/checked:.1%}")
