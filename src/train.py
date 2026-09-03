import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from src.config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, 
    RF_PARAM_DIST, XGB_PARAM_DIST, MODEL_SAVE_PATH
)
from src.preprocess import get_preprocessed_data

def run_training():
    print("Loading and preprocessing data")
    X, y = get_preprocessed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Class imbalance weight for XGBoost
    w = (y_train == 0).sum() / (y_train == 1).sum()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rf_model = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    xgb_model = XGBClassifier(scale_pos_weight=w, random_state=RANDOM_STATE, eval_metric='logloss')

    rf_search = RandomizedSearchCV(
        estimator=rf_model, param_distributions=RF_PARAM_DIST,
        n_iter=15, scoring='roc_auc', cv=skf, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model, param_distributions=XGB_PARAM_DIST,
        n_iter=15, scoring='roc_auc', cv=skf, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )

    print("\nTuning Random Forest")
    rf_search.fit(X_train, y_train)

    print("\nTuning XGBoost:")
    xgb_search.fit(X_train, y_train)

    print("\nFINAL SEARCH RESULTS:")
    print(f"Random Forest Best ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"Random Forest Best Params:  {rf_search.best_params_}\n")
    print(f"XGBoost Best ROC-AUC:       {xgb_search.best_score_:.4f}")
    print(f"XGBoost Best Params:        {xgb_search.best_params_}\n")

    # Extract best models & baseline
    best_rfc = rf_search.best_estimator_
    best_rf_probs = best_rfc.predict_proba(X_test)[:, 1]
    best_rf_preds = best_rfc.predict(X_test)

    best_xgb = xgb_search.best_estimator_
    xgb_probs = best_xgb.predict_proba(X_test)[:, 1]
    xgb_preds = best_xgb.predict(X_test)

    baseline_rfc = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    baseline_rfc.fit(X_train, y_train)
    baseline_rf_probs = baseline_rfc.predict_proba(X_test)[:, 1]
    baseline_rf_preds = baseline_rfc.predict(X_test)

    # Output benchmark reports
    print(" XGBoost Test Performance ")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
    print("\nClassification Report:\n", classification_report(y_test, xgb_preds))

    print(" Tuned RF Test Performance ")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, best_rf_probs):.4f}\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, best_rf_preds))
    print("\nClassification Report:\n", classification_report(y_test, best_rf_preds))

    print(" Baseline RF Test Performance ")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, baseline_rf_probs):.4f}\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_rf_preds))
    print("\nClassification Report:\n", classification_report(y_test, baseline_rf_preds))

    # Persist the winning tuned model
    print(f"\nSaving winning model (Tuned RF) to {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model_data = {"model": best_rfc, "features_names": X.columns.tolist()}
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print("Saved successfully.")

if __name__ == "__main__":
    run_training()