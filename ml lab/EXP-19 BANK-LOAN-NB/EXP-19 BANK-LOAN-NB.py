import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rng = np.random.default_rng(3)
n = 1000
df = pd.DataFrame({
    "Income": rng.normal(5000, 2000, n).clip(1000),
    "LoanAmount": rng.normal(150, 60, n).clip(20),
    "Credit_History": rng.choice([0, 1], n, p=[0.2, 0.8]),
    "Married": rng.choice([0, 1], n),
    "Education": rng.choice([0, 1], n),
    "Dependents": rng.integers(0, 4, n),
})
p = 1 / (1 + np.exp(-(2.5 * df.Credit_History + df.Income / 3000
                      - df.LoanAmount / 100 + 0.3 * df.Education - 1.5)))
df["Loan_Status"] = (rng.random(n) < p).astype(int)

X, y = df.drop(columns="Loan_Status"), df["Loan_Status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

nb = GaussianNB().fit(Xtr, ytr)
pred = nb.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["Rejected", "Approved"]))

applicant = pd.DataFrame([[6000, 120, 1, 1, 1, 0]], columns=X.columns)
print("New applicant ->", "Approved" if nb.predict(applicant)[0] else "Rejected")
