import pickle
import pandas as pd
from src.config import MODEL_SAVE_PATH, ENCODERS_SAVE_PATH
from src.preprocess import clean_data, encode_features

class ChurnPredictor:
    def __init__(self):
        with open(MODEL_SAVE_PATH, "rb") as f:
            model_artifact = pickle.load(f)
        with open(ENCODERS_SAVE_PATH, "rb") as f:
            self.encoders = pickle.load(f)

        self.model = model_artifact["model"]
        self.feature_names = model_artifact.get("features_names") or model_artifact.get("feature_names")

    def predict(self, raw_data: dict, threshold: float = 0.5) -> dict:
        df = pd.DataFrame([raw_data])
        df = clean_data(df)
        df = encode_features(df, fit=False, encoders=self.encoders)
        df = df[self.feature_names]

        churn_probability = float(self.model.predict_proba(df)[0][1])
        is_churn = churn_probability >= threshold

        return {
            "prediction": "Churn" if is_churn else "No Churn",
            "churn_probability": round(churn_probability, 4),
            "risk_status": "High Risk" if churn_probability > 0.7 else "Moderate Risk" if churn_probability > 0.4 else "Low Risk"
        }

if __name__ == "__main__":
    predictor = ChurnPredictor()

    sample_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85
    }

    result = predictor.predict(sample_customer)
    print("\nInference Output:")
    for k, v in result.items():
        print(f"  {k}: {v}")