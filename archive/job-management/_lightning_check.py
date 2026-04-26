import os
os.environ["LIGHTNING_USER_ID"] = "a385c190-699f-4096-9e63-18240c0722e2"
os.environ["LIGHTNING_API_KEY"] = "f4967845-cbb3-48b1-9d6d-8f1a7df683aa"

from lightning_sdk.lightning_cloud.login import Auth
auth = Auth()
auth.authenticate()

import requests, json
r = requests.get("https://lightning.ai/v1/memberships", headers={"Authorization": auth.auth_header}, timeout=10)
print("Status:", r.status_code)
data = r.json()
for m in data.get("memberships", []):
    name = m.get("name", "?")
    username = m.get("username", "?")
    print(f"  name={name}, username={username}")
