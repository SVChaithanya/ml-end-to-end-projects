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
