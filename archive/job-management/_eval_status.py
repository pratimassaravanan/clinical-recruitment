import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Job

job = Job("sft-eval-hf", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Job: {job.name}, Status: {job.status}")
try:
    logs = job.logs
    with open("eval_logs.txt", "w", encoding="utf-8") as f:
        f.write(logs)
    lines = logs.strip().split("\n")
    print(f"\n--- Last 40 lines ({len(lines)} total) ---")
    for line in lines[-40:]:
        print(line)
except Exception as e:
    print(f"Logs: {e}")
