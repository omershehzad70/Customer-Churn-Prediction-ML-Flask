# Customer Churn Prediction Using Machine Learning and Flask

An end-to-end Artificial Intelligence project that predicts whether a telecommunications customer is likely to leave a company using Machine Learning and a Flask-based web application.

This project was developed as part of **Artificial Intelligence Assignments 1 and 2** at the University of Lahore.

## Live Web Application

**[Try the Customer Churn Predictor](https://omershehzad.pythonanywhere.com)**

The deployed application allows users to enter customer information and receive a churn prediction with an estimated probability.

## Project Overview

Customer churn occurs when customers stop using a company's services. Predicting churn can help telecommunications companies identify customers who may be at risk of leaving.

This project uses a **Logistic Regression** machine learning model trained on the Telco Customer Churn dataset.

The application classifies customers into two categories:

- **Churn:** The customer is predicted to leave the company.
- **No Churn:** The customer is predicted to stay with the company.

It also displays the model's estimated churn probability.

## Assignment 1 — Machine Learning Model Training

The first assignment focuses on developing a customer churn prediction model.

The workflow includes:

1. Loading the Telco Customer Churn dataset.
2. Exploring and preprocessing customer data.
3. Preparing numerical and categorical features.
4. Training a Logistic Regression classification model.
5. Evaluating model predictions.
6. Preparing the trained model for integration into a web application.

**Notebook:** `Customer_Churn_Prediction.ipynb`

## Assignment 2 — Flask Web Application

The second assignment extends the machine learning project by developing an interactive web interface using Flask.

### Main Features

- Interactive customer information form
- Customer profile and demographic inputs
- Account and billing information
- Telecommunications service selections
- Machine learning-based churn prediction
- Estimated churn probability
- Color-coded prediction results
- Required-field form validation
- Deployment on PythonAnywhere

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application and machine learning development |
| Pandas | Data handling and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Logistic Regression model |
| Joblib | Saving and loading model artifacts |
| Flask | Backend web framework |
| HTML | Web page structure |
| CSS | User interface styling |
| GitHub | Version control and source code hosting |
| PythonAnywhere | Web application deployment |
| Google Colab | Machine learning notebook development |

## Project Structure

```text
Customer-Churn-Prediction-ML-Flask/
│
├── Customer_Churn_Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn (1).csv
│
├── Assignment-2-Flask-Web-App/
│   ├── app.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── test_app.py
│   ├── requirements.txt
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   ├── README.md
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       └── style.css
│
└── README.md
```

## Running the Application Locally

**1. Clone the repository**

```bash
git clone https://github.com/omershehzad70/Customer-Churn-Prediction-ML-Flask.git
```

**2. Open the Flask application directory**

```bash
cd Customer-Churn-Prediction-ML-Flask/Assignment-2-Flask-Web-App
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the environment**

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
pip install scikit-learn==1.8.0
```

**6. Run the application**

```bash
python app.py
```

Open the local address displayed by Flask in your browser.

## Prediction Examples

The following results were observed during application testing.

| Customer scenario | Prediction | Estimated churn probability |
|---|---|---|
| High-risk customer | Churn | 84.6% |
| Long-term customer | No Churn | 0.5% |
| Another tested customer | No Churn | 11.1% |

These probabilities are model estimates for the tested customer profiles, not guarantees of actual customer behavior.

## Deployment

The Flask application is deployed on **PythonAnywhere** using:

- Python 3.13
- Flask
- Scikit-learn 1.8.0
- A Python virtual environment
- WSGI application configuration
- Static CSS file mapping

**Live URL:** https://omershehzad.pythonanywhere.com

## Future Improvements

Potential improvements include:

- Comparing additional machine learning algorithms
- Improving model evaluation and performance
- Adding customer churn analytics dashboards
- Supporting batch predictions using CSV uploads
- Providing explanations of important prediction factors
- Improving responsive design

## Academic Information

**Project:** Customer Churn Prediction Using Machine Learning and Flask

**Student:** Omer Shehzad

**University:** University of Lahore

**Course:** Artificial Intelligence

**Assignments:** 1 — Model Training, 2 — Flask Web Application

## Conclusion

This project demonstrates the complete workflow of a machine learning application, from preparing customer data and training a classification model to integrating the trained model into a Flask web application and deploying it online.

The resulting system provides an accessible interface for estimating telecommunications customer churn risk.
