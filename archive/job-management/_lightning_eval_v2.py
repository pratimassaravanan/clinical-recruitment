import os, time
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine, Job

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# Upload fixed eval script
s.upload_file("_eval_remote_v2.py", remote_path="eval_v2.py")
print("Script uploaded")

# Submit as job
job = Job.run(
    command="pip install httpx -q && python eval_v2.py 2>&1",
    name="sft-eval-v2",
    machine=Machine.L4,
    studio=s,
)
print(f"Job: {job.name}")
print(f"Link: {job.link}")

# Poll
while True:
    status = job.status
    print(f"  {status} ({time.strftime('%H:%M:%S')})")
    if str(status) in ("Status.Completed", "Status.Failed", "Status.Stopped"):
        break
    time.sleep(30)

print(f"\nFinal: {job.status}")
try:
    logs = job.logs
    with open("eval_logs_v2.txt", "w", encoding="utf-8") as f:
        f.write(logs)
    lines = logs.strip().split("\n")
    for line in lines[-30:]:
        print(line)
except Exception as e:
    print(f"Logs: {e}")

s.stop()
print("Studio stopped")
