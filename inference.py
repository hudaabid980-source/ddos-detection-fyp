"""Prediction logic for the DDoS detector (no Streamlit imports, so it can be unit-tested)."""
import numpy as np
import pandas as pd

BENIGN_LABEL = "BENIGN"


def prepare_features(df: pd.DataFrame, required_features):
    """Return (X, missing_columns). X is numeric, finite and in the exact training column order."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()   # CICDDoS2019 headers sometimes have stray spaces
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        return None, missing
    X = df[list(required_features)].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, []


def predict(model, encoder, X: pd.DataFrame) -> pd.DataFrame:
    """Predict on RAW features. The Random Forest was trained on unscaled data, so NO scaler is used."""
    proba = model.predict_proba(X)
    class_ids = model.classes_[proba.argmax(axis=1)]
    labels = encoder.inverse_transform(class_ids.astype(int))
    out = pd.DataFrame({
        "Prediction": labels,
        "Confidence": proba.max(axis=1).round(4),
    }, index=X.index)
    out["Traffic Type"] = np.where(out["Prediction"] == BENIGN_LABEL, "Benign", "Attack")
    return out
