"""Re-creates the Assignment 1 final model (Logistic Regression) and saves
the model, scaler and feature list for the Flask app."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import encode

BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df_model = encode(df, drop_first=True)
X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
print("F1-Score :", round(f1_score(y_test, pred), 4))

joblib.dump(model, os.path.join(BASE, 'model.joblib'))
joblib.dump(scaler, os.path.join(BASE, 'scaler.joblib'))
joblib.dump(list(X.columns), os.path.join(BASE, 'feature_columns.joblib'))
print("\nSaved: model.joblib, scaler.joblib, feature_columns.joblib")
print("Feature count:", len(X.columns))
