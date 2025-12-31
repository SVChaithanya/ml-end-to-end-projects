from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
import joblib
import pandas as pd
import logging
import psycopg2
import os 

logging.basicConfig(level=logging.INFO)

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

app = FastAPI(title="Credit Risk Prediction API")

class LoanRequest(BaseModel):
    loan_amnt:float= conint(gt=0)
    annual_inc: float=conint(gt=0)
    dti: float=confloat(ge=0, le=100)
    fico_mean: float=conint(ge=300, le=850)
    int_rate: float=confloat(gt=0)
    term: str
    grade: str
    purpose: str

conn = psycopg2.connect(
    dbname="testdb",
    user="postgres",
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    host="localhost",
    port="****"

)
cursor = conn.cursor()

@app.post("/predict")
def predict_risk(data: LoanRequest):

    input_df = pd.DataFrame([data.dict()])[features]

    probability = model.predict_proba(input_df)[0][1]
    expected_loss = probability * data.loan_amnt

    if expected_loss < 5000:
        risk = "Low"
    elif expected_loss < 15000:
        risk = "Medium"
    else:
        risk = "High"

    cursor.execute(
        """
        INSERT INTO predictions 
        (loan_amnt, probability, expected_loss, risk_level)
        VALUES (%s, %s, %s, %s)
        """,
        (data.loan_amnt, probability, expected_loss, risk)
    )
    conn.commit()

    logging.info("Prediction stored in database")

    return {
        "probability": round(probability, 3),
        "expected_loss": round(expected_loss, 2),
        "risk_level": risk
    }

