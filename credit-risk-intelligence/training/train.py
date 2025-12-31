import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df=pd.read_csv("loan.csv")
df['fico_mean'] = (df['fico_range_low'] + df['fico_range_high']) / 2
bad_status = [
    'Charged Off',
    'Default',
    'Late (31-120 days)',
    'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off'
]

good_status = [
    'Fully Paid',
    'Current',
    'In Grace Period',
    'Does not meet the credit policy. Status:Fully Paid'
]

df = df[df['loan_status'].isin(bad_status + good_status)]

df['target'] =df['loan_status'].apply(
    lambda x: 1 if x in bad_status else 0)

TARGET = "target"

FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_mean",
    "int_rate",
    "term",
    "grade",
    "purpose"
]

X = df[FEATURES]
y = df[TARGET]

numeric_features = [
    "loan_amnt", "annual_inc", "dti", "fico_mean", "int_rate"
]

categorical_features = [
    "term", "grade", "purpose"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", roc_auc)

joblib.dump(pipeline, "model.pkl")
joblib.dump(FEATURES, "features.pkl")
joblib.dump({"roc_auc": roc_auc}, "metrics.pkl")

print("Model and preprocessing saved successfully")

#ROC curve in lgbm

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of the LightGBM model')
plt.legend(loc="lower right")
plt.show()

