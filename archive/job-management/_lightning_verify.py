import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
if str(s.status) == "Status.Stopped":
    s.start(machine=Machine.CPU)
    import time; time.sleep(15)

# Download trainer_state from best checkpoint
os.makedirs("lightning_output/checkpoints", exist_ok=True)

# The best checkpoint (load_best_model_at_end=True) — check which one
output = s.run("cat /teamspace/jobs/sft-train-l40s/artifacts/train_output/sft/checkpoint-1809/trainer_state.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({\"best_metric\": d.get(\"best_metric\"), \"best_model_checkpoint\": d.get(\"best_model_checkpoint\"), \"epoch\": d.get(\"epoch\"), \"total_flos\": d.get(\"total_flos\")}, indent=2))'")
print("Best checkpoint info:", output)

# Also check if there's a special_tokens_map.json anywhere
output2 = s.run("find /teamspace/jobs/sft-train-l40s/artifacts/train_output/ -name 'special_tokens_map.json' 2>/dev/null || echo 'none found'")
print("special_tokens_map:", output2)

# Stop
s.stop()
print("Done")
