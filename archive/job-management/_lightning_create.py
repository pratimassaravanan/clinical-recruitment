import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602", create_ok=True)
print(f"Studio: {s.name}, status: {s.status}")
s.start(machine=Machine.A100)
print(f"Started! Status: {s.status}")
