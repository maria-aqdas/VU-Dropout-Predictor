import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

df = pd.read_csv("/home/claude/vu-dropout/vu_dropout_dataset.csv")
features = [
    "weekly_lms_logins", "video_watch_pct", "avg_quiz_score",
    "assignment_completion_rate", "avg_submission_delay_days",
    "forum_posts_per_month", "session_attendance_pct",
    "gpa_trend", "current_gpa", "part_time_job", "semester"
]
X = df[features].values
y = df["dropout_risk"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic Regression - chosen for explainability (coefficients export cleanly to JS)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
print(confusion_matrix(y_test, lr_pred))

# Random Forest - for comparison / feature importance sanity check
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))

importances = dict(zip(features, rf.feature_importances_.round(4)))
print("\nFeature importances (Random Forest):")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Export logistic regression weights + scaler params for pure-JS inference in the frontend
export = {
    "features": features,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0]),
    "feature_importance_rf": {k: float(v) for k, v in importances.items()},
    "test_accuracy": float((lr_pred == y_test).mean()),
    "test_roc_auc": float(roc_auc_score(y_test, lr_proba)),
}
with open("/home/claude/vu-dropout/model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nExported model_export.json for frontend inference.")
