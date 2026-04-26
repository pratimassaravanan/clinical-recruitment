import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# Find where the LoRA files actually are
print("\n--- Finding files ---")
output = s.run("find /teamspace/jobs/sft-train-l40s/ -name '*.safetensors' -o -name 'adapter_config.json' -o -name 'tokenizer.json' 2>/dev/null | head -20")
print("Job artifacts:", output)

output2 = s.run("find /teamspace/studios/this_studio/train_output/ -type f 2>/dev/null | head -30")
print("Studio train_output:", output2)

# Check the job snapshot/artifacts paths
output3 = s.run("ls -la /teamspace/jobs/sft-train-l40s/ 2>/dev/null")
print("Job dir:", output3)

output4 = s.run("ls -la /teamspace/jobs/sft-train-l40s/artifacts/ 2>/dev/null || echo 'no artifacts dir'")
print("Artifacts:", output4)

output5 = s.run("ls -la /teamspace/jobs/sft-train-l40s/snapshot/ 2>/dev/null || echo 'no snapshot dir'")
print("Snapshot:", output5)

# Check if model was saved in the snapshot
output6 = s.run("find /teamspace/jobs/sft-train-l40s/snapshot/ -name '*.safetensors' -o -name 'adapter_config.json' 2>/dev/null | head -20")
print("Snapshot models:", output6)
