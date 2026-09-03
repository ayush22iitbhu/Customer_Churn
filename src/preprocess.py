import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_PATH, ENCODERS_SAVE_PATH

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes customerID and casts TotalCharges to numeric."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df

def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """Encodes categorical fields, persisting encoders during training."""
    df = df.copy()
    
    # Target encoding if present
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    object_cols = df.select_dtypes(include="object").columns.tolist()

    if fit:
        encoders = {}
        for col in object_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        os.makedirs(os.path.dirname(ENCODERS_SAVE_PATH), exist_ok=True)
        with open(ENCODERS_SAVE_PATH, "wb") as f:
            pickle.dump(encoders, f)
        return df, encoders
    else:
        if encoders is None:
            with open(ENCODERS_SAVE_PATH, "rb") as f:
                encoders = pickle.load(f)
        for col in object_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
        return df

def get_preprocessed_data():
    """Pipeline entry for training data."""
    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = clean_data(raw_df)
    processed_df, encoders = encode_features(cleaned_df, fit=True)
    X = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]
    return X, y