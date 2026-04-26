#!/usr/bin/env python3
"""Launch full REINFORCE v3 run with fixed observation parsing."""
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

# Upload the fixed REINFORCE script
s.upload_file('_lightning_reinforce.py', remote_path='reinforce_v3_fixed.py')
print('Uploaded reinforce_v3_fixed.py')

# Launch job
job = Job.run(
    command='python reinforce_v3_fixed.py 2>&1',
    name='rl-v3-fixed',
    machine=Machine.L40S,
    studio=s,
)
print('Job:', job.name)
print('Status:', job.status)
print(f'Link: https://lightning.ai/pratimassaravanan/inference-optimization-project/studios/rl-v2/app?app_id=jobs&job_name={job.name}')
