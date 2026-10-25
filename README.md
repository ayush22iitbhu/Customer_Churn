# Telecom Customer Churn Prediction

An end-to-end Machine Learning pipeline that identifies at-risk subscribers using the **Telco Customer Churn** dataset (7,043 customer records).
[Google Colab](https://colab.research.google.com/drive/13HCn9lB9RALsojUOMEQpXJT7TkhBPMPl)


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


## Model Benchmark & Evaluation

All evaluations were executed on an untouched 20% holdout test set ($N = 1,409$).

### Comparative Performance

| Model Configuration | Churn Recall | Churn Precision | Missed Churners (FN) | False Alarms (FP) | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline Random Forest** | 0.47 | 0.67 | 197 | 86 | 0.8364 |
| **Tuned XGBoost (`best_xgb`)** | 0.82 | 0.52 | 68 | 279 | 0.8597 |
| **Tuned Random Forest (`best_rfc`)** ⭐| 0.86 | 0.53 | 54 | 284 | 0.8626 |

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
