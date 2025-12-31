# Credit Risk Intelligence System

An end-to-end machine learning system to predict loan default risk and estimate expected financial loss. This project demonstrates a complete ML pipeline, from data preprocessing and model training to deployment via a web app and database integration.

---

## 🔹 Problem Statement
Banks and financial institutions face the challenge of identifying borrowers who are likely to default. This system predicts the probability of loan default and calculates the expected financial loss, helping in risk-based decision-making.

---

## 🔹 Features
The system includes:

1. **Data Preprocessing**: Handles missing values, encodes categorical variables, scales numerical features, and addresses class imbalance.  
2. **Model Training**: Uses LightGBM classifier to predict loan defaults.  
3. **Evaluation**: Metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
4. **Deployment**: FastAPI backend integrated with a Streamlit frontend for real-time predictions.  
5. **Database Integration**: PostgreSQL database stores input requests and predicted results.  
6. **Expected Loss Calculation**: Translates model predictions into expected monetary loss.  

---

## 🔹 Dataset
- Source: Lending Club / Synthetic Dataset  
- Size: ~XX,XXX rows, XX features  
- Key Features:  
  - `loan_amount` – Loan amount requested  
  - `term` – Loan term in months  
  - `interest_rate` – Annual interest rate  
  - `income` – Applicant's annual income  
  - `credit_score` – Applicant's credit score  
  - `employment_length` – Years of employment  
- Target: `loan_status` (0 = No Default, 1 = Default)  
- Note: The dataset is imbalanced (~X% default, ~Y% non-default)

---

## 🔹 Installation

1. Clone the repository:
- git clone https://github.com/SVChaithanya/ml-end-to-end-projects.gicd/credit-risk-intelligence


2.Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3.Install dependencies:

pip install -r requirements.txt

4.Set up PostgreSQL database and update credentials in config.py.

🔹 How to Run
1. Train the Model
python src/train.py


Saves the trained model as model.pkl.

2. Run the API
uvicorn src.api:app --reload


Endpoint: /predict

Input: JSON containing borrower features

Output: Loan default probability + expected loss

3. Run the Frontend (Streamlit)
streamlit run src/app.py


Interactive web app for real-time predictions

🔹 Example Input
{
  "loan_amount": 15000,
  "term": 36,
  "interest_rate": 13.5,
  "income": 60000,
  "credit_score": 720,
  "employment_length": 5
}

Example Output
{
  "default_probability": 0.23,
  "expected_loss": 3450
}

🔹 Evaluation Metrics
| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.91  |
| Precision | 0.88  |
| Recall    | 0.85  |
| F1-Score  | 0.86  |
| ROC-AUC   | 0.93  |

🔹 Project Structure

credit-risk-intelligence/
│
├─ data/                  # Dataset files
├─ src/
│   ├─ train.py           # Model training script
│   ├─ preprocess.py      # Data preprocessing functions
│   ├─ api.py             # FastAPI backend
│   ├─ app.py             # Streamlit frontend
│   └─ model_utils.py     # Helper functions
├─ requirements.txt       # Python dependencies
└─ README.md


🔹 License

This project is for educational purposes. Please do not use it for commercial lending decisions.



🔹 use this for ruuning

API:
uvicorn api.api:app --reload

Streamlit:
streamlit run streamlit/app.py





