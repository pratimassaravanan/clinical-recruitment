import os, time
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine, Job

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# Upload files
print("Uploading 10K traces (25MB)...")
s.upload_file("data/sft_traces_10k.json", remote_path="sft_traces.json")
print("Traces uploaded!")

print("Uploading pipeline script...")
s.upload_file("_lightning_full_pipeline.py", remote_path="train.py")
print("Script uploaded!")

# Submit job on L40S
job = Job.run(
    command="pip install httpx huggingface_hub -q && python train.py 2>&1",
    name="sft-10k-v2",
    machine=Machine.L40S,
    studio=s,
)
print(f"Job submitted: {job.name}")
print(f"Link: {job.link}")

# Poll
while True:
    status = job.status
    print(f"  {status} ({time.strftime('%H:%M:%S')})")
    if str(status) in ("Status.Completed", "Status.Failed", "Status.Stopped"):
        break
    time.sleep(60)

print(f"\nFinal: {job.status}")
try:
    logs = job.logs
    with open("pipeline_logs.txt", "w", encoding="utf-8") as f:
        f.write(logs)
    lines = logs.strip().split("\n")
    for line in lines[-40:]:
        print(line)
except Exception as e:
    print(f"Logs: {e}")

s.stop()
print("Studio stopped")
