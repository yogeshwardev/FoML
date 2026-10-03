import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
X = 2 * np.random.rand(100, 1) * 5
y = 3 * X.ravel() + 4 + np.random.randn(100)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print(f"Slope = {model.coef_[0]:.3f}, Intercept = {model.intercept_:.3f}")
print("MSE:", round(mean_squared_error(yte, pred), 4))
print("R2 :", round(r2_score(yte, pred), 4))

plt.scatter(Xtr, ytr, label="train")
plt.scatter(Xte, yte, c="g", label="test")
plt.plot(X, model.predict(X), "r", label="fit")
plt.legend(); plt.title("Linear Regression"); plt.show()
