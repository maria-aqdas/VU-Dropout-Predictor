# Vital Signs — Distance-Learning Dropout Risk Predictor

A data science project that predicts dropout risk for distance-learning students using engagement signals (LMS logins, quiz scores, assignment completion, GPA trend, and more).

**Live demo:** _(add your Vercel link here after deploying)_

## Why this project

Distance-learning dropout rates are high, and unlike on-campus programs, warning signs often go unnoticed until a student has already disengaged. This project explores whether early engagement data — logins, quiz performance, submission timing — can flag at-risk students before that happens.

## How it works

- `generate_data.py` — generates a synthetic dataset of 3,000 student-terms, built to reflect correlations reported in real distance-education dropout research (since real, identifiable student records aren't accessible for a project like this).
- `train_model.py` — trains and evaluates a Logistic Regression model (82% test accuracy, ROC-AUC 0.85) and a Random Forest for feature-importance comparison. Exports the trained weights to `model_export.json`.
- `index.html` — the interactive dashboard. The exported model weights are embedded directly in the page, so risk scoring runs **entirely client-side in JavaScript** — no backend server, which keeps hosting free.

## Top predictors found by the model

1. GPA trend (declining vs. improving)
2. Assignment completion rate
3. Video lecture watch percentage
4. Average quiz score

## Tech stack

Python (pandas, scikit-learn) for data generation and modeling · vanilla HTML/CSS/JS for the frontend · deployed on Vercel.

## Limitations

The dataset is synthetic, not real student records — a production version would need an anonymized data-sharing agreement with a university. This project demonstrates the full pipeline (data → model → deployed product) rather than claiming real-world validated results.
