"""
train.py
--------
Credit Card Approval Prediction - Model Training Pipeline

Steps:
 1. Load applicant data (application_record + credit_record merged).
 2. Feature engineering:
      - Convert DAYS_BIRTH / DAYS_EMPLOYED into AGE_YEARS / EMPLOYED_YEARS
      - Convert multi-class past-due payment status into a binary
        high-risk flag (as required by Scenario 2 - compliance review)
      - One-hot encode categorical fields
 3. Train/test split
 4. Train four classifiers:
      - Logistic Regression
      - Random Forest
      - XGBoost (Gradient Boosting)
      - Decision Tree
 5. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC
 6. Save the best-performing model + preprocessing pipeline to model/model.pkl
 7. Save a comparison chart and metrics report
"""

import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("data/credit_card_approval.csv")

# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
df["EMPLOYED_YEARS"] = np.where(
    df["DAYS_EMPLOYED"] == 365243, 0, (-df["DAYS_EMPLOYED"] / 365).round(1)
)

# Scenario 2: convert multi-class past-due payment status into a binary
# high-risk label. Any applicant with 2+ months past-due history is
# flagged high risk and treated as ineligible signal for the model.
df["HIGH_RISK_FLAG"] = (df["PAST_DUE_COUNT"] >= 2).astype(int)

binary_map = {"Y": 1, "N": 0, "M": 1, "F": 0}
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map(binary_map)
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map(binary_map)
df["CODE_GENDER"] = df["CODE_GENDER"].map(binary_map)

categorical_cols = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

feature_cols = [c for c in df_encoded.columns if c not in ("ID", "APPROVED", "DAYS_BIRTH", "DAYS_EMPLOYED")]
X = df_encoded[feature_cols]
y = df_encoded["APPROVED"]

numeric_cols = [
    "AMT_INCOME_TOTAL", "AGE_YEARS", "EMPLOYED_YEARS", "CNT_CHILDREN",
    "CNT_FAM_MEMBERS", "PAST_DUE_COUNT", "CREDIT_INQUIRIES_6M",
    "EXISTING_LOANS", "ACCOUNT_AGE_MONTHS",
]

# ---------------------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------------------------------------------------------------
# 4. Train four classifiers
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
}

results = {}
trained_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    trained_models[name] = model
    print(f"{name:22s} | Acc: {results[name]['accuracy']:.4f} | "
          f"F1: {results[name]['f1_score']:.4f} | "
          f"AUC: {results[name]['roc_auc']:.4f}")

# ---------------------------------------------------------------------
# 5. Pick best model (by F1 score)
# ---------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1_score"])
best_model = trained_models[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------------
# 6. Save model bundle (model + scaler + feature list + metadata)
# ---------------------------------------------------------------------
bundle = {
    "model": best_model,
    "model_name": best_name,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "requires_scaling": best_name == "Logistic Regression",
    "metrics": results[best_name],
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(bundle, f)

with open("model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved model bundle -> model/model.pkl")
print("Saved metrics report -> model/metrics.json")

# ---------------------------------------------------------------------
# 7. Comparison chart
# ---------------------------------------------------------------------
names = list(results.keys())
f1s = [results[n]["f1_score"] for n in names]
accs = [results[n]["accuracy"] for n in names]
aucs = [results[n]["roc_auc"] for n in names]

x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, accs, width, label="Accuracy", color="#4C72B0")
ax.bar(x, f1s, width, label="F1 Score", color="#DD8452")
ax.bar(x + width, aucs, width, label="ROC-AUC", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10)
ax.set_ylim(0, 1)
ax.set_title("Model Comparison - Credit Card Approval Prediction")
ax.legend()
plt.tight_layout()
plt.savefig("model/model_comparison.png", dpi=150)
print("Saved comparison chart -> model/model_comparison.png")

# Confusion matrix of the best model
if best_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig2, ax2 = plt.subplots(figsize=(4.5, 4))
im = ax2.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Rejected", "Approved"])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Rejected", "Approved"])
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
ax2.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
print("Saved confusion matrix -> model/confusion_matrix.png")
