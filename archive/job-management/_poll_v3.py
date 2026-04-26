"""Poll rl-v3-fixed job until complete."""
import os, time, sys
os.environ['LIGHTNING_USER_ID'] = '1b95df13-0074-4163-9734-1c344df3605f'
os.environ['LIGHTNING_API_KEY'] = 'f0987f67-a55f-4b26-ba62-821384a36a48'
from lightning_sdk.lightning_cloud.login import Auth
auth = Auth()
auth.authenticate()
from lightning_sdk import Job

for i in range(30):
    j = Job(name='rl-v3-fixed', teamspace='inference-optimization-project', user='pratimassaravanan')
    s = j.status
    c = j.total_cost
    print(f"[poll {i}] status={s} cost=${c:.2f}", flush=True)
    if s in ("Completed", "Failed", "Stopped"):
        print(f"DONE: {s}")
        try:
            print(j.logs[-3000:])
        except Exception as e:
            print(f"logs err: {e}")
        sys.exit(0)
    time.sleep(60)

print("Timeout after 30 polls")
