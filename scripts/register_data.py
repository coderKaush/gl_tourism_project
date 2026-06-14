"""register_data.py — GitHub Actions: register-dataset job.
Creates HF dataset repo and uploads tourism.csv."""
import os
from huggingface_hub import HfApi, login

login(token=os.environ['HF_TOKEN'])
api = HfApi()

HF_USER      = 'k0t0320d'
DATASET_REPO = f'{HF_USER}/tourism-wellness-data'

api.create_repo(repo_id=DATASET_REPO, repo_type='dataset', exist_ok=True)
print(f'Dataset repo: https://huggingface.co/datasets/{DATASET_REPO}')

api.upload_file(
    path_or_fileobj='data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=DATASET_REPO,
    repo_type='dataset',
)
print('tourism.csv registered on Hugging Face Hub.')
