import os, time
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine, Job

# Use existing studio as the environment
s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# Submit as a Job on L40S (48GB VRAM — no offloading, 3x faster than L4)
# The job uses the studio's environment (files already uploaded)
machines_to_try = [
    ("L40S", Machine.L40S),
    ("L4", Machine.L4),
    ("T4", Machine.T4),
]

job = None
for name, machine in machines_to_try:
    try:
        job = Job.run(
            command="python train.py 2>&1",
            name=f"sft-train-{name.lower()}",
            machine=machine,
            studio=s,
        )
        print(f"Job submitted on {name}! Name: {job.name}")
        print(f"Link: {job.link}")
        break
    except Exception as e:
        print(f"  {name}: {str(e)[:80]}")

if job:
    # Poll for completion
    print("\nWaiting for job to complete...")
    while True:
        status = job.status
        print(f"  Status: {status}")
        if str(status) in ("Status.Completed", "Status.Failed", "Status.Stopped"):
            break
        time.sleep(30)
    
    print(f"\nFinal status: {job.status}")
    try:
        print(f"Logs:\n{job.logs}")
    except:
        print("(Logs not yet available)")
