import os
os.environ['LIGHTNING_USER_ID'] = '1b95df13-0074-4163-9734-1c344df3605f'
os.environ['LIGHTNING_API_KEY'] = 'f0987f67-a55f-4b26-ba62-821384a36a48'
from lightning_sdk.lightning_cloud.login import Auth
auth = Auth()
auth.authenticate()
from lightning_sdk import Job

for name in ["rl-v3-fixed", "rl-v3", "debug-trial-53ujn"]:
    try:
        j = Job(name=name, teamspace="inference-optimization-project", user="pratimassaravanan")
        status = j.status
        cost = j.total_cost
        print(f"{name}: status={status} cost=${cost:.2f}")
        if status in ("Completed", "Stopped", "Failed"):
            try:
                logs = j.logs()
                lines = logs.strip().split('\n') if logs else []
                print(f"  Log lines: {len(lines)}")
                # Print last 30 lines
                for line in lines[-30:]:
                    print(f"  | {line}")
            except Exception as e:
                print(f"  Logs error: {e}")
    except Exception as e:
        print(f"{name}: error={e}")
