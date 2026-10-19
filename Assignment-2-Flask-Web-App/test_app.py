"""Local tests: web predictions vs. the saved model used directly."""
import re
import joblib, pandas as pd
from app import app, feature_columns, scaler, model
from preprocess import build_features

CHURN_CASE = {
    'gender': 'Female', 'SeniorCitizen': '0', 'Partner': 'No', 'Dependents': 'No',
    'tenure': '2', 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': '85.50', 'TotalCharges': '171.00'}
STAY_CASE = {
    'gender': 'Male', 'SeniorCitizen': '0', 'Partner': 'Yes', 'Dependents': 'Yes',
    'tenure': '60', 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
    'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Two year', 'PaperlessBilling': 'No',
    'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': '65.00', 'TotalCharges': '3900.00'}

def direct_model(case):
    row = dict(case); row['SeniorCitizen'] = int(row['SeniorCitizen'])
    for k in ('tenure', 'MonthlyCharges', 'TotalCharges'): row[k] = float(row[k])
    X = scaler.transform(build_features(pd.DataFrame([row]), feature_columns))
    return int(model.predict(X)[0]), float(model.predict_proba(X)[0][1])

def label_from_html(html):
    m = re.search(r'result-label">([^<]+)<', html)
    p = re.search(r'probability: ([\d.]+)%', html)
    return (m.group(1) if m else None), (p.group(1) if p else None)

c = app.test_client()
print("GET / ->", c.get('/').status_code)

for name, case in [("EXPECTED CHURN", CHURN_CASE), ("EXPECTED NO-CHURN", STAY_CASE)]:
    r = c.post('/predict', data=case)
    web_label, web_prob = label_from_html(r.get_data(as_text=True))
    pred, prob = direct_model(case)
    print(f"\n{name}: HTTP {r.status_code}")
    print("  Website shows :", web_label, f"({web_prob}%)")
    print("  Model directly:", 'Churn' if pred else 'No Churn', f"({prob*100:.1f}%)")
    print("  Consistent    :", web_label == ('Churn' if pred else 'No Churn'))

print("\n--- Error handling ---")
bad = dict(STAY_CASE); bad['tenure'] = ''
r = c.post('/predict', data=bad); print("Empty tenure       -> HTTP", r.status_code, "|", re.findall(r'<li>(.*?)</li>', r.get_data(as_text=True)))
bad = dict(STAY_CASE); bad['tenure'] = 'abc'
r = c.post('/predict', data=bad); print("Text in number     -> HTTP", r.status_code, "|", re.findall(r'<li>(.*?)</li>', r.get_data(as_text=True)))
bad = dict(STAY_CASE); bad['MonthlyCharges'] = '-5'
r = c.post('/predict', data=bad); print("Negative charges   -> HTTP", r.status_code, "|", re.findall(r'<li>(.*?)</li>', r.get_data(as_text=True)))
bad = dict(STAY_CASE); bad['Contract'] = 'Lifetime'
r = c.post('/predict', data=bad); print("Fake dropdown value-> HTTP", r.status_code, "|", re.findall(r'<li>(.*?)</li>', r.get_data(as_text=True)))
bad = dict(STAY_CASE); bad['tenure'] = 'nan'
r = c.post('/predict', data=bad); print("'nan' typed        -> HTTP", r.status_code, "|", re.findall(r'<li>(.*?)</li>', r.get_data(as_text=True)))
r = c.post('/predict', data={}); print("Completely empty   -> HTTP", r.status_code, "|", len(re.findall(r'<li>', r.get_data(as_text=True))), "messages, no crash")
