import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "Annual_Income": rng.normal(50000, 15000, n).clip(10000),
    "Num_Bank_Accounts": rng.integers(1, 10, n),
    "Num_Credit_Cards": rng.integers(1, 10, n),
    "Interest_Rate": rng.integers(2, 30, n),
    "Num_Delayed_Payments": rng.integers(0, 25, n),
    "Credit_Utilization": rng.uniform(10, 60, n),
    "Outstanding_Debt": rng.uniform(0, 5000, n),
})
score = (df.Annual_Income / 10000 - df.Interest_Rate / 5 - df.Num_Delayed_Payments / 4
         - df.Outstanding_Debt / 1000 - df.Credit_Utilization / 10 + rng.normal(0, 1, n))
df["Credit_Score"] = pd.qcut(score, 3, labels=["Poor", "Standard", "Good"])

X, y = df.drop(columns="Credit_Score"), df["Credit_Score"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)
pred = model.predict(sc.transform(Xte))
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred))

new = pd.DataFrame([[60000, 4, 3, 8, 2, 25, 500]], columns=X.columns)
print("New customer credit score:", model.predict(sc.transform(new))[0])
