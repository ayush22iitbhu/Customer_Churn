# Telecom Customer Churn Prediction & Retention Engine

An end-to-end Machine Learning pipeline that identifies at-risk subscribers using the **Telco Customer Churn** dataset (7,043 customer records). Built with Scikit-Learn and XGBoost, this system shifts the optimization objective from standard accuracy to **minority class recall** and **ROC-AUC**, reducing undetected churners by **72.6%**.

---

## Business Problem & Impact

* **High Acquisition Costs:** In the telecommunications sector, acquiring a new subscriber costs 5–7x more than retaining an existing one.
* **The Imbalance Trap:** The dataset exhibits a 73.4% to 26.6% class imbalance. A naive model predicting "No Churn" for every customer scores ~73% accuracy while capturing 0% of churners.
* **Objective:** Prioritize **Recall (Class 1)** and **ROC-AUC** to catch churners before they exit, accepting a calculated trade-off in false alarms to protect recurring revenue.

---

## Architecture & Tech Stack

* **Core Stack:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Data Cleaning & Pipeline:**
  * Resolved empty whitespace strings (`" "`) in `TotalCharges` and cast to float.
  * Encoded categorical attributes using `LabelEncoder` and persisted mappings via `encoders.pkl` to prevent inference drift.
* **Cross-Validation:** 5-fold `StratifiedKFold` ensuring consistent ~73:27 class distribution across all evaluation folds.
* **Cost-Sensitive Learning:**
  * Random Forest: `class_weight='balanced'`
  * XGBoost: `scale_pos_weight=2.78` (ratio of negative to positive instances)
* **Hyperparameter Tuning:** 15-iteration `RandomizedSearchCV` scored directly on `roc_auc`.

---

## Model Benchmark & Evaluation

All evaluations were executed on an untouched 20% holdout test set ($N = 1,409$).

### Comparative Performance

| Model Configuration | Churn Recall | Churn Precision | Missed Churners (FN) | False Alarms (FP) | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline Random Forest** | 0.47 | 0.67 | 197 | 86 | 0.8364 |
| **Tuned XGBoost (`best_xgb`)** | 0.82 | 0.52 | 67 | 280 | 0.8597 |
| **Tuned Random Forest (`best_rfc`)** ⭐| 0.86 | 0.52 | 54 | 284 | 0.8626 |

---

### Selected Model: Tuned Random Forest (`best_rfc`)

```text
Confusion Matrix (Test Set: 1,409 samples):
               Predicted: Stay    Predicted: Churn
Actual: Stay        752 (TN)            284 (FP)
Actual: Churn        54 (FN)            319 (TP)

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.73      0.82      1036
           1       0.53      0.86      0.65       373
           1       0.53      0.86      0.65       373
