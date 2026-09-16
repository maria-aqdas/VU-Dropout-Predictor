import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "vu_dropout_dataset.csv"
EXPORT_PATH = "model_export.json"

RANDOM_STATE = 42

FEATURES = [
    "weekly_lms_logins",
    "video_watch_pct",
    "avg_quiz_score",
    "assignment_completion_rate",
    "avg_submission_delay_days",
    "forum_posts_per_month",
    "session_attendance_pct",
    "gpa_trend",
    "current_gpa",
    "part_time_job",
    "semester"
]

TARGET = "dropout_risk"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("RETAINIQ — MACHINE LEARNING PIPELINE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. DATA VALIDATION
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isnull().sum().sum() > 0:
    print("\nMissing values detected:")
    print(df[required_columns].isnull().sum())
else:
    print("\n✓ No missing values")


if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError(
        "Target column must contain only 0 and 1."
    )

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget distribution (%):")
print(
    (df[TARGET].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 4. FEATURES / TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. MODELS
# ============================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    class_weight=None,
    n_jobs=-1
)


# ============================================================
# 7. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


def evaluate_cv(model, name):

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print(f"\n{name} — 5-Fold Cross Validation")

    for metric in scoring:
        values = results[f"test_{metric}"]

        print(
            f"{metric:10s}: "
            f"{values.mean():.4f} "
            f"(± {values.std():.4f})"
        )

    return results


lr_cv = evaluate_cv(
    logistic_model,
    "Logistic Regression"
)

rf_cv = evaluate_cv(
    random_forest_model,
    "Random Forest"
)


# ============================================================
# 8. TRAIN FINAL MODELS
# ============================================================

logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


# ============================================================
# 9. TEST SET EVALUATION
# ============================================================

lr_probability = logistic_model.predict_proba(
    X_test
)[:, 1]

rf_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]

lr_prediction = (
    lr_probability >= 0.50
).astype(int)

rf_prediction = (
    rf_probability >= 0.50
).astype(int)


def evaluate_model(
    name,
    y_true,
    prediction,
    probability
):

    cm = confusion_matrix(
        y_true,
        prediction
    )

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability
        )
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            prediction,
            zero_division=0
        )
    )

    return metrics, cm


lr_metrics, lr_cm = evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_prediction,
    lr_probability
)

rf_metrics, rf_cm = evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_prediction,
    rf_probability
)


# ============================================================
# 10. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    prediction = (
        lr_probability >= threshold
    ).astype(int)

    threshold_results.append({

        "threshold": round(
            float(threshold),
            2
        ),

        "precision": round(
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        )
    })


for row in threshold_results:

    print(
        f"Threshold={row['threshold']:.2f} | "
        f"Precision={row['precision']:.3f} | "
        f"Recall={row['recall']:.3f} | "
        f"F1={row['f1']:.3f}"
    )


# ============================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance = dict(
    zip(
        FEATURES,
        random_forest_model.feature_importances_
    )
)

rf_importance = {
    key: round(float(value), 4)
    for key, value in rf_importance.items()
}


print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

for feature, importance in sorted(
    rf_importance.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{feature:35s}: "
        f"{importance:.4f}"
    )


# ============================================================
# 12. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

lr_model = logistic_model.named_steps["model"]
scaler = logistic_model.named_steps["scaler"]

coefficients = dict(
    zip(
        FEATURES,
        lr_model.coef_[0]
    )
)

coefficients = {
    key: round(float(value), 6)
    for key, value in coefficients.items()
}


# ============================================================
# 13. ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    lr_probability
)

roc_points = []

for i in range(
    min(len(fpr), 50)
):

    roc_points.append({

        "fpr": round(
            float(fpr[i]),
            6
        ),

        "tpr": round(
            float(tpr[i]),
            6
        )
    })


# ============================================================
# 14. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        lr_probability
    )
)

pr_points = []

for i in range(
    min(len(precision_curve), 50)
):

    pr_points.append({

        "precision": round(
            float(precision_curve[i]),
            6
        ),

        "recall": round(
            float(recall_curve[i]),
            6
        )
    })


# ============================================================
# 15. EXPORT MODEL
# ============================================================

export = {

    "project": "RetainIQ",

    "dataset": {
        "rows": int(df.shape[0]),
        "features": len(FEATURES),
        "positive_cases": int(y.sum()),
        "positive_rate": float(y.mean())
    },

    "features": FEATURES,

    "target": TARGET,

    "model": {
        "type": "LogisticRegression",
        "reason": "Explainable probability-based risk scoring"
    },

    "scaler": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },

    "logistic_regression": {

        "coefficients": (
            lr_model.coef_[0].tolist()
        ),

        "intercept": float(
            lr_model.intercept_[0]
        )
    },

    "random_forest": {

        "feature_importance": rf_importance
    },

    "test_metrics": {

        "accuracy": lr_metrics["accuracy"],
        "precision": lr_metrics["precision"],
        "recall": lr_metrics["recall"],
        "f1": lr_metrics["f1"],
        "roc_auc": lr_metrics["roc_auc"],
        "pr_auc": lr_metrics["pr_auc"]
    },

    "confusion_matrix": {

        "true_negative": int(lr_cm[0][0]),
        "false_positive": int(lr_cm[0][1]),
        "false_negative": int(lr_cm[1][0]),
        "true_positive": int(lr_cm[1][1])
    },

    "threshold_analysis": threshold_results,

    "roc_curve": roc_points,

    "precision_recall_curve": pr_points,

    "evaluation_note": (
        "Performance is based on a synthetic dataset "
        "and should not be interpreted as validated "
        "real-world student dropout prediction performance."
    )
}


with open(
    EXPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        export,
        file,
        indent=2
    )


print("\n" + "=" * 70)
print("✓ Model export created:")
print(EXPORT_PATH)
print("=" * 70)
