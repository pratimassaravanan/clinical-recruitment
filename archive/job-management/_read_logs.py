import os
os.environ['LIGHTNING_USER_ID'] = '1b95df13-0074-4163-9734-1c344df3605f'
os.environ['LIGHTNING_API_KEY'] = 'f0987f67-a55f-4b26-ba62-821384a36a48'
from lightning_sdk.lightning_cloud.login import Auth
auth = Auth()
auth.authenticate()
from lightning_sdk import Job

for name in ["debug-trial-53ujn", "rl-v3"]:
    try:
        j = Job(name=name, teamspace="inference-optimization-project", user="pratimassaravanan")
        print(f"\n{'='*60}")
        print(f"JOB: {name} status={j.status} cost=${j.total_cost:.2f}")
        print(f"{'='*60}")
        logs = j.logs  # property, not method
        if logs:
            lines = logs.strip().split('\n')
            print(f"Total lines: {len(lines)}")
            for line in lines[-60:]:
                print(line)
        else:
            print("No logs")
    except Exception as e:
        print(f"{name}: error={e}")
