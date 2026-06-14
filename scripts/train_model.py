"""train_model.py — GitHub Actions: model-training job.
Loads processed splits from HF, runs GridSearchCV on 6 classifiers,
logs all runs to MLflow, serialises and registers the best model."""
import os, json, pickle
import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from huggingface_hub import hf_hub_download, HfApi, login

login(token=os.environ['HF_TOKEN'])
api = HfApi()

HF_USER      = 'k0t0320d'
DATASET_REPO = f'{HF_USER}/tourism-wellness-data'
MODEL_REPO   = f'{HF_USER}/tourism-wellness-model'
TARGET       = 'ProdTaken'

tr_path = hf_hub_download(repo_id=DATASET_REPO, filename='train.csv', repo_type='dataset')
te_path = hf_hub_download(repo_id=DATASET_REPO, filename='test.csv',  repo_type='dataset')
tr_df = pd.read_csv(tr_path); te_df = pd.read_csv(te_path)
X_tr = tr_df.drop(columns=[TARGET]); y_tr = tr_df[TARGET]
X_te = te_df.drop(columns=[TARGET]); y_te = te_df[TARGET]
print(f'Train: {X_tr.shape}  |  Test: {X_te.shape}')

CANDIDATE_MODELS = {
    'DecisionTree':     (DecisionTreeClassifier(class_weight='balanced', random_state=0),
                         {'max_depth': [4, 6, None], 'min_samples_leaf': [1, 5, 10]}),
    'Bagging':          (BaggingClassifier(random_state=0),
                         {'n_estimators': [20, 50, 100], 'max_samples': [0.6, 0.8, 1.0]}),
    'RandomForest':     (RandomForestClassifier(class_weight='balanced', random_state=0),
                         {'n_estimators': [100, 200], 'max_depth': [6, 10, None]}),
    'AdaBoost':         (AdaBoostClassifier(random_state=0),
                         {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.5, 1.0]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=0),
                         {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]}),
    'XGBoost':          (XGBClassifier(scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                                       random_state=0, eval_metric='logloss'),
                         {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6]}),
}

mlflow.set_experiment('WellnessTourism_PurchasePrediction')
top_f1 = 0.0; best_clf_name = None; best_clf = None
results = []

for name, (est, grid) in CANDIDATE_MODELS.items():
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(est, grid, cv=5, scoring='f1', n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        y_pred = gs.best_estimator_.predict(X_te)
        y_prob = gs.best_estimator_.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics({'test_accuracy': acc, 'test_f1': f1, 'test_roc_auc': auc})
        print(f'{name}: Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}')
        results.append({'model': name, 'accuracy': acc, 'f1_score': f1, 'roc_auc': auc})
        if f1 > top_f1:
            top_f1 = f1; best_clf_name = name; best_clf = gs.best_estimator_

print(f'\nBest model: {best_clf_name}  (F1={top_f1:.4f})')

os.makedirs('tourism_project/model_building', exist_ok=True)
MODEL_PKL = 'tourism_project/model_building/best_model.pkl'
FEAT_JSON = 'tourism_project/model_building/feature_columns.json'
with open(MODEL_PKL, 'wb') as fh: pickle.dump(best_clf, fh)
with open(FEAT_JSON, 'w') as fh: json.dump(list(X_tr.columns), fh)

api.create_repo(repo_id=MODEL_REPO, repo_type='model', exist_ok=True)
for lp, hn in [(MODEL_PKL, 'best_model.pkl'), (FEAT_JSON, 'feature_columns.json')]:
    api.upload_file(path_or_fileobj=lp, path_in_repo=hn,
                    repo_id=MODEL_REPO, repo_type='model')
    print(f'Registered {hn} -> https://huggingface.co/{MODEL_REPO}')
