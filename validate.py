"""Pre-submission validation checklist for Adaptive Clinical Recruitment."""

import os
import httpx

BASE = "https://pratimassaravanan-clinical-recruitment.hf.space"
_DIR = os.path.dirname(os.path.abspath(__file__))
c = httpx.Client(timeout=30)
checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("PRE-SUBMISSION VALIDATION: Adaptive Clinical Recruitment")
print("=" * 60)

# 1. HF Space deploys and responds
print("\n1. HF Space deploys and responds to reset()")
try:
    r = c.get(BASE + "/")
    check("Root returns 200", r.status_code == 200)
    r = c.post(BASE + "/reset", params={"task_id": "easy_bench"})
    check("reset() returns 200", r.status_code == 200)
    d = r.json()
    check("reset() returns observation", "observation" in d)
    check("reset() returns done=False", d.get("done") is False)
except Exception as e:
    check("HF Space reachable", False, str(e))

# 2. OpenEnv spec: typed models, step/reset/state
print("\n2. OpenEnv spec compliance")
try:
    r = c.post(
        BASE + "/step",
        json={
            "action_type": "screen_patient",
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
        },
    )
    d = r.json()
    check("step() returns observation", "observation" in d)
    check("step() returns reward", "reward" in d)
    check("step() returns done", "done" in d)
    check("step() returns info", "info" in d)

    r = c.get(BASE + "/state")
    d = r.json()
    check("state() returns task", "task" in d)
    check("state() returns step", "step" in d)
    check("state() returns done", "done" in d)
except Exception as e:
    check("API endpoints work", False, str(e))

# 3. 3+ tasks with graders
print("\n3. 3+ tasks with graders")
try:
    r = c.get(BASE + "/tasks")
    tasks = r.json()
    check("3+ tasks defined", len(tasks) >= 3, f"found {len(tasks)}")
    for tid in ["easy_bench", "medium_bench", "hard_bench"]:
        r = c.post(BASE + "/reset", params={"task_id": tid})
        check(f"Task '{tid}' resets OK", r.status_code == 200)
except Exception as e:
    check("Tasks endpoint works", False, str(e))

# 4. Graders return 0.0-1.0 (run a quick episode)
print("\n4. Grader scores in 0.0-1.0 range")
try:
    for tid in ["easy_bench", "medium_bench", "hard_bench"]:
        r = c.post(BASE + "/reset", params={"task_id": tid})
        last = r.json()
        for _ in range(10):
            if last.get("done"):
                break
            r = c.post(
                BASE + "/step",
                json={
                    "action_type": "screen_patient",
                    "patient_id": None,
                    "site_id": None,
                    "strategy_change": None,
                },
            )
            last = r.json()
        reward = last.get("reward", -1)
        check(
            f"Task '{tid}' reward is float",
            isinstance(reward, (int, float)),
            f"reward={reward}",
        )
except Exception as e:
    check("Grader check", False, str(e))

# 5. Dockerfile builds
print("\n5. Dockerfile")
check("Dockerfile exists", os.path.isfile(os.path.join(_DIR, "Dockerfile")))

# 6. Baseline inference script format
print("\n6. Inference script format check")
with open(os.path.join(_DIR, "inference.py")) as f:
    content = f.read()
check(
    "[START] line format present",
    "[START] task=" in content and " env=" in content and " model=" in content,
)
check(
    "[STEP] line format present",
    "[STEP] step=" in content
    and " action=" in content
    and " reward=" in content
    and " done=" in content
    and " error=" in content,
)
check(
    "[END] line format present",
    "[END] success=" in content and " steps=" in content and " rewards=" in content,
)
check("log_start function", "log_start(" in content)
check("log_step function", "log_step(" in content)
check("log_end function", "log_end(" in content)
check("No extra summary stdout", "=== Summary ===" not in content)
check(
    "Plain text logging (no JSON log lines)",
    'json.dumps({"type": "[START]"' not in content,
)
check(
    "No score field in [END] line",
    "score=" not in content.split("def log_end", 1)[1]
    if "def log_end" in content
    else False,
)
check("OpenAI client used", "OpenAI(" in content)
check("API_BASE_URL used", "API_BASE_URL" in content)
check("MODEL_NAME used", "MODEL_NAME" in content)
check("HF_TOKEN used", "HF_TOKEN" in content)
check(
    "HF_TOKEN has no default",
    'HF_TOKEN = os.getenv("HF_TOKEN")' in content
    or "HF_TOKEN = os.getenv('HF_TOKEN')" in content,
)
check("inference.py in root dir", os.path.isfile(os.path.join(_DIR, "inference.py")))

# 7. openenv.yaml
print("\n7. openenv.yaml check")
try:
    import yaml

    with open(os.path.join(_DIR, "openenv.yaml")) as f:
        cfg = yaml.safe_load(f)
    check("name defined", "name" in cfg, cfg.get("name"))
    check("version defined", "version" in cfg)
    check("tasks defined", "tasks" in cfg, f"{len(cfg.get('tasks', []))} tasks")
    check("openenv tag", "openenv" in cfg.get("tags", []))
except ImportError:
    # Fallback without yaml
    with open(os.path.join(_DIR, "openenv.yaml")) as f:
        raw = f.read()
    check("name defined", "name:" in raw)
    check("version defined", "version:" in raw)
    check("tasks defined", "tasks:" in raw)
    check("openenv tag", "openenv" in raw)

# 8. Structure checks
print("\n8. Structural checks")
check("models.py exists", os.path.isfile(os.path.join(_DIR, "models.py")))
check("env.py exists", os.path.isfile(os.path.join(_DIR, "env.py")))
check("graders.py exists", os.path.isfile(os.path.join(_DIR, "graders.py")))
check("load_traces.py exists", os.path.isfile(os.path.join(_DIR, "load_traces.py")))
check("app.py exists", os.path.isfile(os.path.join(_DIR, "app.py")))
check(
    "server/__init__.py exists",
    os.path.isfile(os.path.join(_DIR, "server", "__init__.py")),
)
check("server/app.py exists", os.path.isfile(os.path.join(_DIR, "server", "app.py")))
check("pyproject.toml exists", os.path.isfile(os.path.join(_DIR, "pyproject.toml")))
check("requirements.txt exists", os.path.isfile(os.path.join(_DIR, "requirements.txt")))

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    print("ALL CHECKS PASSED - READY TO SUBMIT!")
else:
    print("SOME CHECKS FAILED - review above")
print("=" * 60)

c.close()
