# Credit Risk Intelligence System

End-to-end machine learning system for loan default prediction and expected loss estimation.

This project includes:
- LightGBM model training
- FastAPI backend
- Streamlit frontend
- PostgreSQL database integration
# Credit Risk Intelligence System

This project implements an end-to-end machine learning system for credit risk assessment.

## Overview
- Predicts probability of loan default using LightGBM
- Computes expected financial loss
- Exposes predictions through a FastAPI service
- Provides a Streamlit-based user interface
- Stores predictions in a PostgreSQL database

## Project Structure
training/train.py     -> Model training and evaluation  
api/api.py            -> FastAPI prediction service  
streamlit/app.py      -> Streamlit user interface  
data/sample_loan.csv  -> Sample dataset format  

## How to Run
1. Train the model:
   python training/train.py

2. Start API:
   uvicorn api.api:app --reload

3. Run Streamlit UI:
   streamlit run streamlit/app.py

## Output
- Default probability
- Expected loss
- Risk level (Low / Medium / High)


final : # Credit Risk Intelligence System

An end-to-end machine learning system for predicting loan default risk and estimating expected financial loss. This project demonstrates a complete ML pipeline, from data preprocessing and model training to real-time inference and user interaction.

---

## Project Overview

Credit risk assessment is a critical task in financial decision-making. This system predicts the probability of loan default using a LightGBM model and translates it into expected loss to support practical risk-based decisions. The trained model is deployed through a FastAPI backend and accessed via a Streamlit user interface, with prediction results stored in a PostgreSQL database.

---

## Project Components

- **Model Training**: LightGBM classifier with preprocessing and feature engineering  
- **Backend API**: FastAPI service for real-time prediction  
- **Frontend UI**: Streamlit application for user input and result visualization  
- **Database**: PostgreSQL for storing prediction logs and audit data  

---

## Folder Structure

credit-risk-intelligence/
├── training/
│ └── train.py
├── api/
│ └── api.py
├── streamlit/
│ └── app.py
├── data/
│ └── sample_loan.csv
├── requirements.txt
└── README.md

yaml
Copy code

---

## Dataset

The model was trained on a large historical loan dataset containing borrower information, credit history, and loan attributes.  
Due to size and licensing constraints, only a small representative sample dataset (`sample_loan.csv`) is included in this repository for demonstration purposes.

---

## How to Run the Project

### 1. Install Dependencies

pip install -r requirements.txt
2. Train the Model
python training/train.py
This step trains the LightGBM model and saves the trained pipeline for inference.

3. Start the FastAPI Server
uvicorn api.api:app --reload
The API exposes a /predict endpoint that accepts loan details and returns default probability, expected loss, and risk category.

4. Run the Streamlit Application
streamlit run streamlit/app.py
The Streamlit UI allows users to input borrower details and view prediction results interactively.

Sample API Input:
{
  "loan_amnt": 15000,
  "annual_inc": 48000,
  "dti": 16.3,
  "fico_mean": 650,
  "int_rate": 10.9,
  "term": "36 months",
  "grade": "B",
  "purpose": "home_improvement"
}
Sample Output:
{
  "probability": 0.182,
  "expected_loss": 2734.5,
  "risk_level": "Low"
}
Risk Level Definition
Low Risk: Expected Loss < 5,000

Medium Risk: 5,000 ≤ Expected Loss < 15,000

High Risk: Expected Loss ≥ 15,000
