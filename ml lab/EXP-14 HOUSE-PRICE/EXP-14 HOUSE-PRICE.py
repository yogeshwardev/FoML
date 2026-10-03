from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = fetch_california_housing()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name:18s} R2 = {r2_score(yte, p):.4f}  RMSE = {mean_squared_error(yte, p) ** 0.5:.4f}")

print("\nFeature importances (Random Forest):")
for f, i in sorted(zip(data.feature_names, models["Random Forest"].feature_importances_),
                   key=lambda t: -t[1]):
    print(f"  {f:12s} {i:.3f}")
