import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(1)
n = 800
df = pd.DataFrame({
    "Year": rng.integers(2005, 2024, n),
    "Present_Price": rng.uniform(3, 40, n),
    "Kms_Driven": rng.integers(2000, 150000, n),
    "Fuel_Type": rng.choice(["Petrol", "Diesel", "CNG"], n),
    "Transmission": rng.choice(["Manual", "Automatic"], n),
    "Owner": rng.integers(0, 3, n),
})
df["Selling_Price"] = (df.Present_Price * 0.7 - (2024 - df.Year) * 0.4
                       - df.Kms_Driven / 40000 - df.Owner * 0.5
                       + (df.Fuel_Type == "Diesel") * 1.5
                       + (df.Transmission == "Automatic") * 1.0
                       + rng.normal(0, 0.5, n)).clip(0.3)

df = pd.get_dummies(df, columns=["Fuel_Type", "Transmission"], drop_first=True)
X, y = df.drop(columns="Selling_Price"), df["Selling_Price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

for name, m in (("Linear Regression", LinearRegression()),
                ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=0))):
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name:18s} R2 = {r2_score(yte, p):.4f}  MAE = {mean_absolute_error(yte, p):.4f}")
