import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
X = np.sort(6 * np.random.rand(80, 1) - 3, axis=0)
y = 0.5 * X.ravel() ** 3 - X.ravel() ** 2 + 2 + np.random.randn(80)

lin = LinearRegression().fit(X, y)
poly = PolynomialFeatures(degree=3)
Xp = poly.fit_transform(X)
pr = LinearRegression().fit(Xp, y)

for name, pred in (("Linear", lin.predict(X)), ("Polynomial (deg 3)", pr.predict(Xp))):
    print(f"{name:20s} MSE = {mean_squared_error(y, pred):.4f}   R2 = {r2_score(y, pred):.4f}")

plt.scatter(X, y, label="data")
plt.plot(X, lin.predict(X), "r", label="Linear")
plt.plot(X, pr.predict(Xp), "g", label="Polynomial")
plt.legend(); plt.title("Linear vs Polynomial Regression"); plt.show()
