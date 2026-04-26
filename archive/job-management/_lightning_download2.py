import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")

# Create local dirs
os.makedirs("lightning_output/lora_adapter", exist_ok=True)

# Copy from job artifacts to studio working dir first, then download
print("Copying job artifacts to studio...")
s.run("cp -r /teamspace/jobs/sft-train-l40s/artifacts/train_output/lora_adapter/ /teamspace/studios/this_studio/lora_download/")
s.run("ls -la /teamspace/studios/this_studio/lora_download/")

# List all files in the lora adapter
output = s.run("ls -la /teamspace/studios/this_studio/lora_download/")
print("Files:", output)

# Download each file
files_to_download = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
]

print("\nDownloading LoRA adapter...")
for fname in files_to_download:
    remote = f"lora_download/{fname}"
    local = f"lightning_output/lora_adapter/{fname}"
    try:
        s.download_file(remote, local)
        size = os.path.getsize(local)
        print(f"  {fname}: {size:,} bytes")
    except Exception as e:
        print(f"  {fname}: skipped ({e})")

# Also get the training logs
print("\nGetting training logs...")
try:
    log_output = s.run("cat /teamspace/jobs/sft-train-l40s/artifacts/.zsh_history 2>/dev/null || echo 'no history'")
    print("History:", log_output[:500])
except:
    pass

# Get eval results from the script output
print("\nGetting eval results from training output...")
eval_output = s.run("find /teamspace/jobs/sft-train-l40s/artifacts/ -name '*.json' -not -path '*.cache*' 2>/dev/null | head -20")
print("JSON files:", eval_output)

# Stop studio to save money
print("\nStopping studio...")
s.stop()
print("Studio stopped. Done!")
