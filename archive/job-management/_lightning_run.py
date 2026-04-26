import os, time
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}, machine: {s.machine}")

# Upload files
print("\nUploading traces (8MB)...")
s.upload_file("data/sft_traces_turns_small.json", remote_path="sft_traces.json")
print("Traces uploaded!")

print("Uploading training script...")
s.upload_file("_lightning_train.py", remote_path="train.py")
print("Script uploaded!")

# Install deps and run
print("\nInstalling dependencies...")
s.run("pip install unsloth -q")
s.run("pip install --no-deps 'trl>=0.19.0' -q")
s.run("pip install 'transformers>=5.2.0,<=5.5.0' -q")
s.run("pip install 'datasets>=2.21.0' 'accelerate>=0.34.0' httpx -q")
print("Dependencies installed!")

print("\nStarting training...")
output = s.run("python train.py 2>&1")
print(output)
