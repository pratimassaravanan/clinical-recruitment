import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# Stop the 8xA100 — way too expensive
if s.status.value != "stopped":
    print("Stopping 8xA100...")
    s.stop()
    print("Stopped")

# Restart on L4 (24GB VRAM, ~$1/hr, plenty fast)
try:
    s.start(machine=Machine.L4)
    print(f"Started on L4! Status: {s.status}")
except Exception as e:
    print(f"L4 failed: {e}")
    # Try T4 as fallback
    try:
        s.start(machine=Machine.T4)
        print(f"Started on T4! Status: {s.status}")
    except Exception as e2:
        print(f"T4 also failed: {e2}")
        # Just use whatever was available
        s.start(machine=Machine.A100_80GB_X_8)
        print(f"Back to 8xA100: {s.status}")
