import sys

required = [
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'joblib'
]

try:
    import dask.dataframe as dd
    has_dask = True
except Exception:
    has_dask = False

try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

# %% [markdown]
# Imports

# %%
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# %% [markdown]
# Helper functions

# %%
import warnings
warnings.filterwarnings('ignore')

def find_label_column(df, ncheck=1000):
    candidates = []
    for col in df.columns:
        vals = pd.Series(df[col].dropna().astype(float).unique()[:5]) if hasattr(df[col], 'unique') else None
        if vals is None:
            continue
        if set(vals.astype(int).unique()).issubset({0,1}):
            candidates.append(col)
    return candidates[0] if candidates else None

# %% [markdown]
# Load dataset (auto-detect gz and header)

# %%
path_opts = ['higgs_kaggle.csv', 'higgs_kaggle.csv.gz', 'HIGGS.csv', 'HIGGS.csv.gz', 'higgs.csv', 'higgs.csv.gz']
path = None
for p in path_opts:
    if os.path.exists(p):
        path = p
        break
if path is None:
    raise FileNotFoundError('Put the Kaggle Higgs CSV in the working dir and name it one of: ' + ','.join(path_opts))

if has_dask:
    df = dd.read_csv(path, header=0)
    sample = df.head(20000)
else:
    if path.endswith('.gz'):
        df = pd.read_csv(path, compression='gzip', header=0)
    else:
        df = pd.read_csv(path, header=0)
    sample = df.sample(min(len(df), 20000), random_state=42)

label_col = find_label_column(sample)
if label_col is None:
    for guess in ['label', 'Label', 'target', 'Target', 'y', 'class', 'Class']:
        if guess in (df.columns.tolist() if not has_dask else df.columns):
            label_col = guess
            break
if label_col is None:
    raise ValueError('Could not detect label column automatically. Rename label column to one of label,target,y,class and retry')

# %% [markdown]
# Prepare features and target. If using dask, convert to pandas sample or to_dask for training with incremental methods

# %%
if has_dask:
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X = X.compute()
    y = y.compute()
else:
    X = df.drop(columns=[label_col])
    y = df[label_col]

# %% [markdown]
# Quick EDA

# %%
print('Rows:', len(X))
print('Columns:', X.shape[1])
print('Label distribution:\n', y.value_counts(normalize=True))

# %% [markdown]
# Train/test split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# Impute simple missing values (median) and scale

# %%
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)

X_train_s = scaler.fit_transform(X_train_imp)
X_test_s = scaler.transform(X_test_imp)

# %% [markdown]
# Optional PCA — toggle if dimensionality reduction desired

# %%
use_pca = False
pca = None
if use_pca:
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_s = pca.fit_transform(X_train_s)
    X_test_s = pca.transform(X_test_s)

# %% [markdown]
# Model 1: Gradient Boosted Trees using XGBoost if available otherwise HistGradientBoosting

# %%
if has_xgb:
    model_xgb = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', n_jobs=8, random_state=42)
    model_xgb.fit(X_train_s, y_train)
else:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model_xgb = HistGradientBoostingClassifier(random_state=42)
    model_xgb.fit(X_train_s, y_train)

# %% [markdown]
# Model 2: Logistic Regression

# %%
model_lr = LogisticRegression(max_iter=1000, n_jobs=8)
model_lr.fit(X_train_s, y_train)

# %% [markdown]
# Model 3: SVM approximated via linear SGDClassifier for scale

# %%
model_svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
model_svm.fit(X_train_s, y_train)

# %% [markdown]
# Evaluation helper

# %%
def eval_model(model, X_t, y_t, name='model'):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_t)[:,1]
    else:
        try:
            probs = model.decision_function(X_t)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
        except Exception:
            probs = model.predict(X_t)
    preds = (probs >= 0.5).astype(int)
    roc = roc_auc_score(y_t, probs)
    pr = average_precision_score(y_t, probs)
    cm = confusion_matrix(y_t, preds)
    print('\nModel:', name)
    print('ROC AUC:', round(roc,4))
    print('PR AUC:', round(pr,4))
    print('Confusion matrix:\n', cm)
    print('\nClassification report:\n', classification_report(y_t, preds, digits=4))
    fpr, tpr, _ = roc_curve(y_t, probs)
    prec, rec, _ = precision_recall_curve(y_t, probs)
    return {'roc':roc, 'pr':pr, 'fpr':fpr, 'tpr':tpr, 'prec':prec, 'rec':rec}

# %% [markdown]
# Evaluate on test set

# %%
res_xgb = eval_model(model_xgb, X_test_s, y_test, name='GradientBoosted')
res_lr = eval_model(model_lr, X_test_s, y_test, name='LogisticRegression')
res_svm = eval_model(model_svm, X_test_s, y_test, name='SVM_SGD')

# %% [markdown]
# Plot ROC and PR curves

# %%
plt.figure()
plt.plot(res_xgb['fpr'], res_xgb['tpr'], label=f'GBT AUC {res_xgb["roc"]:.4f}')
plt.plot(res_lr['fpr'], res_lr['tpr'], label=f'LR AUC {res_lr["roc"]:.4f}')
plt.plot(res_svm['fpr'], res_svm['tpr'], label=f'SVM AUC {res_svm["roc"]:.4f}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('ROC Curves')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(res_xgb['rec'], res_xgb['prec'], label=f'GBT PR AUC {res_xgb["pr"]:.4f}')
plt.plot(res_lr['rec'], res_lr['prec'], label=f'LR PR AUC {res_lr["pr"]:.4f}')
plt.plot(res_svm['rec'], res_svm['prec'], label=f'SVM PR AUC {res_svm["pr"]:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curves')
plt.grid(True)
plt.show()

# %% [markdown]
# Feature importance for tree model

# %%
if has_xgb:
    ft_imp = model_xgb.get_booster().get_score(importance_type='gain')
    items = sorted(ft_imp.items(), key=lambda x: x[1], reverse=True)[:30]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    plt.figure(figsize=(8,6))
    sns.barplot(x=vals, y=names)
    plt.title('Top feature importances (XGBoost)')
    plt.show()
else:
    try:
        vals = model_xgb.feature_importances_
        idx = np.argsort(vals)[-30:][::-1]
        plt.figure(figsize=(8,6))
        sns.barplot(x=vals[idx], y=np.array(X.columns)[idx])
        plt.title('Top feature importances (HistGB)')
        plt.show()
    except Exception:
        pass

# %% [markdown]
# Save models and preprocessing pipeline

# %%
os.makedirs('models', exist_ok=True)
joblib.dump({'imputer': imp, 'scaler': scaler, 'pca': pca}, 'models/preproc.joblib')
joblib.dump(model_xgb, 'models/model_gbt.joblib')
joblib.dump(model_lr, 'models/model_lr.joblib')
joblib.dump(model_svm, 'models/model_svm.joblib')

# %% [markdown]
# Notes

# The notebook uses a robust label detection heuristic. If your dataset has a nonstandard label column, rename it to 'label' or provide the column name manually in the code

