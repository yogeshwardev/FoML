import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(4)
n = 300
df = pd.DataFrame({
    "TV": rng.uniform(0, 300, n),
    "Radio": rng.uniform(0, 50, n),
    "Newspaper": rng.uniform(0, 100, n),
})
df["Sales"] = 3 + 0.045 * df.TV + 0.19 * df.Radio + 0.002 * df.Newspaper + rng.normal(0, 1, n)

X, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

for name, m in (("Linear Regression", LinearRegression()),
                ("Random Forest", RandomForestRegressor(random_state=0))):
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name:18s} R2 = {r2_score(yte, p):.4f}  MAE = {mean_absolute_error(yte, p):.4f}")

lr = LinearRegression().fit(Xtr, ytr)
future = pd.DataFrame([[230, 37, 69], [150, 20, 30]], columns=X.columns)
print("\nFuture ad budgets:\n", future)
print("Predicted sales:", lr.predict(future).round(2))
