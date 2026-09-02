# Telecom Customer Churn Prediction Analytics

An end-to-end Machine Learning system that predicts customer churn on the **Telco Customer Churn** dataset (7,043 records). Built using Scikit-Learn and XGBoost, this project optimizes for **minority class recall** and **ROC-AUC** to identify high-risk churners and mitigate recurring revenue loss.

---

## Business Objective

* **The Problem:** Acquiring a new telecom subscriber costs 5–7x more than retaining an existing one. Unmonitored customer churn directly damages Monthly Recurring Revenue (MRR).
* **The Challenge:** Target class imbalance (~73.4% loyal customers vs. ~26.6% churners). Default accuracy-optimized models achieve superficially high accuracy by simply guessing the majority class, missing most churners.
* **The Goal:** Maximize **Recall** (catching churners) and **ROC-AUC** rather than raw accuracy, drastically minimizing False Negatives (undetected churners).

---

## Tech Stack & Workflow

* **Libraries:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Data Processing & Integrity:** Handled blank string records in `TotalCharges`, cast columns to proper numeric dtypes, and preserved categorical mappings using persisted `LabelEncoder` objects.
* **Validation Strategy:** 5-fold `StratifiedKFold` cross-validation to maintain identical class ratios across every evaluation split.
* **Imbalance Handling:** Algorithmic cost-sensitive reweighting via `class_weight='balanced'` (Random Forest) and `scale_pos_weight=2.78` (XGBoost).
* **Hyperparameter Optimization:** `RandomizedSearchCV` scored against **ROC-AUC** to balance ensemble depth, leaf splits, and learning rates.

---

## Model Performance & Comparison

| Model Architecture | Accuracy | Churn Recall (Caught) | Missed Churners (FN) | False Alarms (FP) | Churn Precision |
|---|---|---|---|---|---|
| Baseline Random Forest (Default) | 80% | 47% (176 / 373) | 197 | **86** | **67%** |
| Default Weighted XGBoost | 77% | 69% (259 / 373) | 114 | 207 | 56% |
| Tuned XGBoost (`best_xgb`) | 75% | 82% (306 / 373) | 67 | 280 | 52% |
| **Tuned Random Forest (`best_rfc`)** ⭐ | **76%** | **86% (319 / 373)** | **54** | **284** | **53%** |

> **Selected Production Model:** **Tuned Random Forest (`best_rfc`)**. It identifies **85.5% of all churning customers** on an untouched test set, reducing missed churners from 197 down to just 54.

---
