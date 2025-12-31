import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Assessment", layout="centered")

st.title("Credit Risk Assessment System")

st.write("Enter borrower details to estimate default risk and expected loss.")

with st.form("loan_form"):

    loan_amnt = st.number_input("Loan Amount", min_value=1000)
    annual_inc = st.number_input("Annual Income", min_value=1000)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 100.0)
    fico_mean = st.slider("FICO Score", 300, 850)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
    term = st.selectbox("Loan Term", ["36 months", "60 months"])
    grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
    purpose = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_mean": fico_mean,
        "int_rate": int_rate,
        "term": term,
        "grade": grade,
        "purpose": purpose
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.metric("Default Probability", result["probability"])
        st.metric("Expected Loss", result["expected_loss"])
        st.metric("Risk Level", result["risk_level"])

        st.caption("Risk level is derived from expected financial loss estimation.")
    else:
        st.error("Prediction failed. Check API.")

