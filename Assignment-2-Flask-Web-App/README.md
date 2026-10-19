# Customer Churn Prediction Web App (Flask)

Web app that takes customer details and predicts **Churn / No Churn** using the
Logistic Regression model from Assignment 1.

## Project structure
```
churn_app/
  app.py                  Flask backend (routes, validation, prediction)
  preprocess.py           Shared encoding code (used by training AND the app)
  train_model.py          Re-trains the model and saves the artifacts
  model.joblib            Saved Logistic Regression model
  scaler.joblib           Saved StandardScaler
  feature_columns.joblib  Exact feature names/order used in training
  requirements.txt
  test_app.py             Local tests
  WA_Fn-UseC_-Telco-Customer-Churn.csv
  templates/index.html    Input form + result display
  static/style.css        Styling
```

## Run locally
```
pip install -r requirements.txt
python train_model.py     # creates model.joblib, scaler.joblib, feature_columns.joblib
python app.py             # open http://127.0.0.1:5000
```

## Deploy on PythonAnywhere
1. Create a free account at pythonanywhere.com.
2. **Files** tab: upload the project ZIP into your home directory, then open a
   **Bash console** and run:
   `unzip churn_app.zip`
3. In the Bash console, create a virtualenv and install packages:
   ```
   mkvirtualenv churn-env --python=python3.10
   cd ~/churn_app
   pip install -r requirements.txt
   python train_model.py
   ```
   (Running `train_model.py` on PythonAnywhere rebuilds the model files with the
   same scikit-learn version the server uses, which avoids version-mismatch errors.)
4. **Web** tab -> *Add a new web app* -> *Manual configuration* -> pick the same
   Python version you used for the virtualenv (e.g. 3.10).
5. On the Web tab set:
   - **Source code:** `/home/YOUR_USERNAME/churn_app`
   - **Virtualenv:** `/home/YOUR_USERNAME/.virtualenvs/churn-env`
6. Click the **WSGI configuration file** link and replace its contents with:
   ```python
   import sys
   path = '/home/YOUR_USERNAME/churn_app'
   if path not in sys.path:
       sys.path.insert(0, path)

   from app import app as application
   ```
7. Click the green **Reload** button. Your site is live at
   `https://YOUR_USERNAME.pythonanywhere.com`.
8. If something fails, check the **Error log** link on the Web tab.
