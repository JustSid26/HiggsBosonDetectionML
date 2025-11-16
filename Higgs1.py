# %% [markdown]
# Higgs Kaggle (Local Files Version)
#
# You provided train.csv (labelled), test.csv (unlabelled), and sample_submission.csv.
# This version automatically loads those local files instead of downloading anything.
#
# It performs:
# - Data loading (train/test)
# - Preprocessing (impute + scale)
# - Model training on train.csv (Gradient Boosted Trees, Logistic Regression, SVM)
# - Prediction on test.csv and generation of a submission file

# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

# %% [markdown]
# Load your local files

# %%
train_path = 'train.csv'
test_path = 'test.csv'
sub_path = 'sample_submission.csv'

assert os.path.exists(train_path), 'train.csv not found'
assert os.path.exists(test_path), 'test.csv not found'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

# %% [markdown]
# Detect label column and prepare features

# %%
label_col = None
for c in train.columns:
    if 'Label' in c or 'label' in c or 'target' in c or 'Target' in c or 'Class' in c:
        label_col = c
        break
if label_col is None:
    label_col = train.columns[-1]  # assume last column

X = train.drop(columns=[label_col])
y = train[label_col].astype(int)
X_test = test.copy()

print('Detected label column:', label_col)
print('Feature count:', X.shape[1])
print('Class balance:\n', y.value_counts(normalize=True))

# %% [markdown]
# Split for validation

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# Preprocessing: impute + scale

# %%
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imp.fit_transform(X_train)
X_val_imp = imp.transform(X_val)
X_test_imp = imp.transform(X_test)

X_train_s = scaler.fit_transform(X_train_imp)
X_val_s = scaler.transform(X_val_imp)
X_test_s = scaler.transform(X_test_imp)

# %% [markdown]
# Model 1: Gradient Boosted Trees (XGBoost if available else HistGradientBoosting)

# %%
if has_xgb:
    model_xgb = xgb.XGBClassifier(objective='binary:logistic', n_jobs=8, tree_method='hist', random_state=42)
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
# Model 3: SVM via SGDClassifier

# %%
model_svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
model_svm.fit(X_train_s, y_train)

# %% [markdown]
# Evaluate

# %%
def evaluate(model, Xv, yv, name):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(Xv)[:,1]
    else:
        try:
            p = model.decision_function(Xv)
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
        except Exception:
            p = model.predict(Xv)
    preds = (p>=0.5).astype(int)
    roc = roc_auc_score(yv, p)
    pr = average_precision_score(yv, p)
    print(f'\n{name}\nROC AUC: {roc:.4f}\nPR AUC: {pr:.4f}')
    print(confusion_matrix(yv, preds))
    print(classification_report(yv, preds, digits=4))
    return roc, pr

res_xgb = evaluate(model_xgb, X_val_s, y_val, 'GradientBoosted')
res_lr = evaluate(model_lr, X_val_s, y_val, 'LogisticRegression')
res_svm = evaluate(model_svm, X_val_s, y_val, 'SVM_SGD')

##ROC COMPARISION
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def get_probs(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:,1]
    try:
        p = model.decision_function(X)
        return (p - p.min()) / (p.max() - p.min() + 1e-12)
    except:
        return model.predict(X)

p_xgb = get_probs(model_xgb, X_val_s)
p_lr = get_probs(model_lr, X_val_s)
p_svm = get_probs(model_svm, X_val_s)

fpr_xgb, tpr_xgb, _ = roc_curve(y_val, p_xgb)
fpr_lr, tpr_lr, _ = roc_curve(y_val, p_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_val, p_svm)

plt.figure(figsize=(8,6))
plt.plot(fpr_xgb, tpr_xgb, label='GradientBoosted')
plt.plot(fpr_lr, tpr_lr, label='LogisticRegression')
plt.plot(fpr_svm, tpr_svm, label='SVM_SGD')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve Comparison')
plt.show()

##precision recall comparision
prec_xgb, rec_xgb, _ = precision_recall_curve(y_val, p_xgb)
prec_lr, rec_lr, _ = precision_recall_curve(y_val, p_lr)
prec_svm, rec_svm, _ = precision_recall_curve(y_val, p_svm)

plt.figure(figsize=(8,6))
plt.plot(rec_xgb, prec_xgb, label='GradientBoosted')
plt.plot(rec_lr, prec_lr, label='LogisticRegression')
plt.plot(rec_svm, prec_svm, label='SVM_SGD')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision Recall Curve Comparison')
plt.show()

# %% [markdown]
# Predictions on test.csv and submission

# %%
if os.path.exists(sub_path):
    submission = pd.read_csv(sub_path)

    # get prediction probability
    if hasattr(model_xgb, "predict_proba"):
        pred = model_xgb.predict_proba(X_test_s)[:, 1]
    else:
        pred = model_xgb.predict(X_test_s)

    # ensure same length
    assert len(pred) == len(submission), "Row mismatch between test.csv and sample_submission.csv"

    # replace prediction column
    pred_col = submission.columns[-1]
    submission[pred_col] = pred

    submission.to_csv("submission.csv", index=False)
    print("submission.csv created and matched with sample_submission.csv")
else:
    print("sample_submission.csv not found.")

# %% [markdown]
# Save models and preprocessing

# 
os.makedirs('models', exist_ok=True)
joblib.dump({'imp':imp,'scaler':scaler},'models/preproc.joblib')
joblib.dump(model_xgb,'models/model_gbt.joblib')
joblib.dump(model_lr,'models/model_lr.joblib')
joblib.dump(model_svm,'models/model_svm.joblib')
print('Models saved in ./models/')
