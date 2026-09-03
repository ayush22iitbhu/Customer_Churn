from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Calculates classification report, confusion matrix, and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n{model_name} Test Performance")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {"accuracy": acc, "roc_auc": roc_auc, "confusion_matrix": cm}