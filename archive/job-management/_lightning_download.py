import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Job, Studio

# Get the job
job = Job("sft-train-l40s", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Job: {job.name}, Status: {job.status}")
print(f"Artifact path: {job.artifact_path}")
print(f"Snapshot path: {job.snapshot_path}")

# Get logs (handle encoding)
try:
    logs = job.logs
    with open("lightning_job_logs.txt", "w", encoding="utf-8") as f:
        f.write(logs)
    print(f"\nLogs saved to lightning_job_logs.txt ({len(logs)} chars)")
    # Print last 80 lines
    lines = logs.strip().split("\n")
    print(f"\n--- Last 80 lines ---")
    for line in lines[-80:]:
        print(line)
except Exception as e:
    print(f"Error getting logs: {e}")

# Download artifacts from the studio
s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"\nStudio: {s.name}, status: {s.status}")

# Start studio to access files if needed
if str(s.status) == "Status.Stopped":
    print("Starting studio to download files...")
    from lightning_sdk import Machine
    s.start(machine=Machine.CPU)
    import time
    time.sleep(10)

# List files in output dir
try:
    print("\nListing remote files...")
    output = s.run("ls -la train_output/lora_adapter/ 2>&1 || echo 'No lora dir'")
    print(output)
    output2 = s.run("ls -la train_output/ 2>&1 || echo 'No output dir'")
    print(output2)
except Exception as e:
    print(f"Error listing: {e}")

# Download the LoRA adapter files
local_dir = "lightning_output/lora_adapter"
os.makedirs(local_dir, exist_ok=True)

adapter_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

print(f"\nDownloading LoRA adapter to {local_dir}/...")
for fname in adapter_files:
    remote = f"train_output/lora_adapter/{fname}"
    local = os.path.join(local_dir, fname)
    try:
        s.download_file(remote, local)
        size = os.path.getsize(local)
        print(f"  {fname}: {size:,} bytes")
    except Exception as e:
        print(f"  {fname}: FAILED - {e}")

# Also try to get the quick eval results
try:
    print("\nChecking for eval output in logs...")
    if logs:
        for line in lines:
            if "JSON parse rate" in line or "Parsed:" in line or "DONE" in line:
                print(f"  {line}")
except:
    pass

print("\nDone!")
