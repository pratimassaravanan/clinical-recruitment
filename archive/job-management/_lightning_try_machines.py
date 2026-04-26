import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk import Studio, Machine

s = Studio(name="clinical-sft", teamspace="deploy-model-project", user="kaushikjpn602")
print(f"Studio: {s.name}, status: {s.status}")

# List all Machine options
machines = [x for x in dir(Machine) if not x.startswith("_") and x[0].isupper()]
print("Available Machine types:", machines)

# Try each
for mname in machines:
    m = getattr(Machine, mname)
    try:
        s.start(machine=m)
        print(f"SUCCESS: {mname} started!")
        break
    except Exception as e:
        err = str(e)[:80]
        print(f"  {mname}: {err}")
