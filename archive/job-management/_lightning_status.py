import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Job

job = Job("sft-train-l40s", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Job: {job.name}")
print(f"Status: {job.status}")
print(f"Machine: {job.machine}")
try:
    logs = job.logs
    # Print last 50 lines
    lines = logs.strip().split("\n")
    for line in lines[-50:]:
        print(line)
except Exception as e:
    print(f"Logs not available yet: {e}")
