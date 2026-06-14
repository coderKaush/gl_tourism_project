"""deploy_app.py — GitHub Actions: deploy-to-space job.
Pushes app.py, requirements.txt, and Dockerfile to HF Docker Space."""
import os
from huggingface_hub import HfApi, login

login(token=os.environ['HF_TOKEN'])
api = HfApi()

HF_USER    = 'k0t0320d'
SPACE_REPO = f'{HF_USER}/tourism-predictor'

api.create_repo(repo_id=SPACE_REPO, repo_type='space', space_sdk='docker', exist_ok=True)
print(f'Space ready -> https://huggingface.co/spaces/{SPACE_REPO}')

DEPLOY_DIR = 'deployment'
for fname in ['app.py', 'requirements.txt', 'Dockerfile']:
    api.upload_file(
        path_or_fileobj=f'{DEPLOY_DIR}/{fname}',
        path_in_repo=fname,
        repo_id=SPACE_REPO,
        repo_type='space',
    )
    print(f'Uploaded {fname}')

print(f'\nApp live at: https://{HF_USER}-tourism-predictor.hf.space')
print('Set Space secret: MODEL_REPO = k0t03n6/tourism-wellness-model')
