"""prepare_data.py — GitHub Actions: data-preparation job.
Loads raw CSV from HF, cleans it, encodes, splits and re-uploads."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi, login

login(token=os.environ['HF_TOKEN'])
api = HfApi()

HF_USER      = 'k0t0320d'
DATASET_REPO = f'{HF_USER}/tourism-wellness-data'
TARGET       = 'ProdTaken'

raw_file = hf_hub_download(repo_id=DATASET_REPO, filename='tourism.csv', repo_type='dataset')
raw_df   = pd.read_csv(raw_file)
print(f'Loaded: {raw_df.shape}')

# Drop non-predictive columns
drop_cols = [c for c in ['Unnamed: 0', 'CustomerID'] if c in raw_df.columns]
raw_df.drop(columns=drop_cols, inplace=True)

# Impute missing values
for col in raw_df.select_dtypes(include=[np.number]).columns:
    raw_df[col] = raw_df[col].fillna(raw_df[col].median())
for col in raw_df.select_dtypes(include='object').columns:
    raw_df[col] = raw_df[col].fillna(raw_df[col].mode()[0])

cat_cols   = raw_df.select_dtypes(include='object').columns.tolist()
encoded_df = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)

X = encoded_df.drop(columns=[TARGET]); y = encoded_df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(f'Train: {X_tr.shape}  |  Test: {X_te.shape}')

os.makedirs('tourism_project/data', exist_ok=True)
tr_data = X_tr.copy(); tr_data[TARGET] = y_tr.values
te_data = X_te.copy(); te_data[TARGET] = y_te.values
tr_data.to_csv('tourism_project/data/train.csv', index=False)
te_data.to_csv('tourism_project/data/test.csv',  index=False)

for lp, hn in [('tourism_project/data/train.csv', 'train.csv'),
               ('tourism_project/data/test.csv',  'test.csv')]:
    api.upload_file(path_or_fileobj=lp, path_in_repo=hn,
                    repo_id=DATASET_REPO, repo_type='dataset')
    print(f'Uploaded {hn}')
print('Data preparation complete.')
