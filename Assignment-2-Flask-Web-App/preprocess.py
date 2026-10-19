"""Shared preprocessing used by BOTH train_model.py and app.py.
Keeping it in one place guarantees the website encodes customer data
exactly the same way the model was trained."""
import pandas as pd

BINARY_YES_NO = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_CATEGORY = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
NUMERIC = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']


def encode(df, drop_first=False):
    """Turn raw customer columns into the numeric format the model expects."""
    out = df.copy()
    for col in BINARY_YES_NO:
        out[col] = out[col].map({'Yes': 1, 'No': 0})
    out['gender'] = out['gender'].map({'Male': 1, 'Female': 0})
    out = pd.get_dummies(out, columns=MULTI_CATEGORY, drop_first=drop_first)
    return out


def build_features(df_raw, feature_columns):
    """Encode raw input and force the exact column order used in training.
    Missing dummy columns are filled with 0; unused ones are dropped."""
    encoded = encode(df_raw, drop_first=False)
    return encoded.reindex(columns=feature_columns, fill_value=0).astype(float)
