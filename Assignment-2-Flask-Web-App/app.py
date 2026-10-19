import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from preprocess import build_features

app = Flask(__name__)

# Absolute paths so the app also works on PythonAnywhere
BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'model.joblib'))
scaler = joblib.load(os.path.join(BASE, 'scaler.joblib'))
feature_columns = joblib.load(os.path.join(BASE, 'feature_columns.joblib'))

# Allowed values for every dropdown field (used for validation)
CHOICES = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': ['0', '1'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)'],
}
# (min, max) for numeric fields
NUMERIC_RANGES = {
    'tenure': (0, 100),
    'MonthlyCharges': (0, 200),
    'TotalCharges': (0, 10000),
}
LABELS = {
    'gender': 'Gender', 'SeniorCitizen': 'Senior citizen', 'Partner': 'Partner',
    'Dependents': 'Dependents', 'PhoneService': 'Phone service',
    'MultipleLines': 'Multiple lines', 'InternetService': 'Internet service',
    'OnlineSecurity': 'Online security', 'OnlineBackup': 'Online backup',
    'DeviceProtection': 'Device protection', 'TechSupport': 'Tech support',
    'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming movies',
    'Contract': 'Contract', 'PaperlessBilling': 'Paperless billing',
    'PaymentMethod': 'Payment method', 'tenure': 'Tenure (months)',
    'MonthlyCharges': 'Monthly charges', 'TotalCharges': 'Total charges',
}


def validate(form):
    """Return (clean_values, errors). Never raises on bad user input."""
    values, errors = {}, []
    for field, options in CHOICES.items():
        v = (form.get(field) or '').strip()
        if not v:
            errors.append(f"Please select a value for '{LABELS[field]}'.")
        elif v not in options:
            errors.append(f"'{v}' is not a valid choice for '{LABELS[field]}'.")
        else:
            values[field] = v
    for field, (lo, hi) in NUMERIC_RANGES.items():
        v = (form.get(field) or '').strip()
        if not v:
            errors.append(f"Please enter a value for '{LABELS[field]}'.")
            continue
        try:
            num = float(v)
        except ValueError:
            errors.append(f"'{LABELS[field]}' must be a number.")
            continue
        if num != num or num < lo or num > hi:   # num != num catches NaN
            errors.append(f"'{LABELS[field]}' must be between {lo} and {hi}.")
        else:
            values[field] = num
    return values, errors


def predict_churn(values):
    """Raw customer values -> (label, churn probability)."""
    row = dict(values)
    row['SeniorCitizen'] = int(row['SeniorCitizen'])
    df_raw = pd.DataFrame([row])
    X = build_features(df_raw, feature_columns)   # same encoding + column order as training
    X_scaled = scaler.transform(X)                # same scaling as training
    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])
    return pred, prob


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', choices=CHOICES, form={}, errors=[], result=None)


@app.route('/predict', methods=['POST'])
def predict():
    values, errors = validate(request.form)
    if errors:
        return render_template('index.html', choices=CHOICES, form=request.form,
                               errors=errors, result=None), 400
    try:
        pred, prob = predict_churn(values)
    except Exception:
        return render_template('index.html', choices=CHOICES, form=request.form,
                               errors=["Something went wrong while making the prediction. "
                                       "Please check your inputs and try again."],
                               result=None), 500
    result = {
        'label': 'Churn' if pred == 1 else 'No Churn',
        'is_churn': pred == 1,
        'probability': round(prob * 100, 1),
        'message': ("This customer is likely to leave. Consider a retention offer."
                    if pred == 1 else
                    "This customer is likely to stay with the company."),
    }
    return render_template('index.html', choices=CHOICES, form=request.form,
                           errors=[], result=result)


if __name__ == '__main__':
    app.run(debug=True)
