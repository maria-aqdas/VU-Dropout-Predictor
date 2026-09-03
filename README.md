# 🩺 Vital Signs

**Early Dropout Risk Prediction for Distance Learners.**

*Predicting student disengagement quietly before it turns into dropout — powered by client-side machine learning.*

---

## 🎯 The Reality

> *"Distance learning dropout doesn't happen with a bang. It happens in silence — skipped videos, late assignments, and zero logins until the exam hall sits empty."*

In remote higher education, student struggle is invisible. Lecturers have hundreds of students across screens, and traditional intervention happens at final grade post-mortem — when recovery is already impossible.

| 🌐 Distance Learning Challenges | 📊 Current Reality |
| --- | --- |
| **Average Global Completion Rate** | Often below **20–30%** in remote/distance models |
| **Drop-off Pattern** | Silent disengagement weeks before formal drop |
| **Intervention Gap** | Usually noticed after failing midterms or missing deadlines |

**Vital Signs** closes that gap. It evaluates 11 behavioral signals in real time, flags students at risk early in the semester, explains *why* the risk exists, and provides concrete intervention suggestions.

---

## 🔁 The Core Decision Loop

$$\text{Behavioral Logs} \longrightarrow \text{Browser ML Scoring} \longrightarrow \text{Explainable Drivers} \longrightarrow \text{Advisor Action} \longrightarrow \text{Outcome Monitoring}$$

1. **Track:** Ingest student engagement signals (LMS logins, video %, assignment completion, quiz performance).
2. **Predict:** Generate instant probability score ($0-100\%$) directly in the client browser.
3. **Explain:** Break down top contributing factors with dynamic feature importance.
4. **Intervene:** Recommend concrete steps (tutoring, deadline restructuring, advisor check-ins).
5. **Simulate:** Run *What If?* analysis to visualize score recovery under changed habits.

---

## 🚀 Key Features

* **Instant Risk Engine:** Interactive sliders calculate risk live without roundtrips to an API server.
* **Interactive Spider / Radar Footprint:** Compares an individual student against cohort baselines across 6 dimensions simultaneously.
* **"What If?" Counterfactual Simulator:** Tests potential interventions (e.g., *"If GPA rises from 1.05 to 2.25, risk drops by 11%"*).
* **Advisor Bulk Audit:** Upload an entire section/cohort via CSV to rank students from highest risk to safest.
* **Fully Client-Side Architecture:** Runs zero-server inference in JavaScript — zero hosting costs, complete student data privacy.
* **Bilingual Accessibility:** Complete system localization in both **English** and **Urdu (اردو)**.
* **Radical Transparency:** Model report card publicly reports false positives, false negatives, precision, and recall alongside honest limitations.

---

## 📊 4 Core Predictive Signals

Based on analysis of 3,000 synthetic student profiles calibrated against empirical distance-learning research:

| Weight | Factor | Why It Predicts Dropout |
| --- | --- | --- |
| **19%** | **GPA Trend (vs. last term)** | Velocity matters more than static GPA; rapid downward trajectory signals burn-out. |
| **15%** | **Assignment Completion %** | Strongest indicator of active academic responsibility and weekly effort. |
| **15%** | **Video Lectures Watched %** | Direct proxy for self-directed study routine in asynchronous programs. |
| **12%** | **Average Quiz Score** | Early formative assessment check indicating foundational comprehension. |

---

## 📈 Model Performance & Honest Limitations

Trust in academic machine learning requires showing failures alongside successes.

| Metric | Score | Detail |
| --- | --- | --- |
| **Overall Accuracy** | **83%** | Evaluated on holdout test set (600 students) |
| **ROC-AUC** | **0.85** | Strong class separability across threshold sweeps |
| **Precision** | **73%** | When flagged high risk, the student is genuinely struggling in 3 of 4 cases |
| **Recall** | **46%** | Catches roughly half of total dropouts |
| **F1 Score** | **56%** | Harmonic balance between precision and recall |

### Confusion Matrix (600 Test Students)

|  | Predicted Safe | Predicted At-Risk |
| --- | --- | --- |
| **Actually Safe** | **428** (True Negative) | **25** (False Alarm) |
| **Actually Dropped Out** | **80** (Missed / False Negative) | **67** (Caught At-Risk) |

> ⚠️ **Honest Limitation:** With a **46% recall**, Vital Signs is strictly an advisor-assistive decision support tool — not an automated gatekeeper. It must be paired with advisor empathy, student surveys, and proactive communication.

---

## ⚙️ Tech Stack & Architecture

```text
├── Data Pipeline (Python / scikit-learn / Pandas)
│   └── 3,000 Cohort Simulation ➔ Feature Engineering ➔ Logistic/Ensemble Modeling
└── Client Interface (HTML5 / Vanilla CSS3 / JavaScript)
    ├── Inline Matrix Vector Multiplication (No Server API)
    ├── Chart.js Dynamic Visualizations (Radar, Bars, ROC Curves)
    └── Local Storage & CSV Parser

```

* **Modeling & Verification:** Python, Pandas, scikit-learn
* **Frontend:** Vanilla JavaScript (ES6+), Semantic HTML5, CSS3 Variables
* **Data Visualizations:** Chart.js
* **Deployment:** Vercel edge deployment

---

## 🧪 Quickstart & Local Setup

Clone the repository:

```bash
git clone https://github.com/your-username/vital-signs.git
cd vital-signs

```

Run locally:

```bash
# Vital Signs runs purely on standard client-side web tech.
# Open directly in your browser:
open index.html

# Or serve via lightweight HTTP server:
npx serve .

```

---

## 🗺️ Roadmap

* [x] Client-side real-time calculation pipeline
* [x] English / Urdu localization toggle
* [x] Radar chart individual vs. cohort benchmarking
* [x] CSV cohort batch processing
* [ ] Exportable PDF summary for academic counseling sessions
* [ ] University LMS (Moodle / Canvas) API webhook integration
* [ ] Longitudinal tracking across multi-semester cohort histories

---

## 👩‍💻 Author

**Maria Aqdas**

Data Science Student · Distance Learner

* [LinkedIn](https://www.google.com/search?q=https%3A%2F%2Fwww.linkedin.com%2F)
* [GitHub](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2F)
* [Email](https://www.google.com/search?q=mailto%3Amariaaqdas.vu%40gmail.com)
