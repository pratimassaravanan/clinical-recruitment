#!/usr/bin/env python3
"""Launch DEBUG TRIAL on Lightning AI — 3 episodes, 15 steps, max debug output."""
import os

os.environ['LIGHTNING_USER_ID'] = '1b95df13-0074-4163-9734-1c344df3605f'
os.environ['LIGHTNING_API_KEY'] = 'f0987f67-a55f-4b26-ba62-821384a36a48'

from lightning_sdk.lightning_cloud.login import Auth
auth = Auth()
auth.authenticate()
print("Authenticated:", auth.user_id)

from lightning_sdk import Studio, Machine, Job

s = Studio(name='rl-v2', teamspace='inference-optimization-project', user='pratimassaravanan')
print('Studio:', s.name, 'status:', s.status)

# Upload the debug trial script
s.upload_file('_debug_trial.py', remote_path='debug_trial.py')
print('Uploaded debug_trial.py')

# Launch job
job = Job.run(
    command='python debug_trial.py 2>&1',
    name='debug-trial',
    machine=Machine.L40S,
    studio=s,
)
print('Job:', job.name)
print('Status:', job.status)
print(f'Link: https://lightning.ai/pratimassaravanan/inference-optimization-project/studios/rl-v2/app?app_id=jobs&job_name=debug-trial')
