import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

model = LogisticRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print("Coefficients:", model.coef_, "Intercept:", model.intercept_)
print("Accuracy:", accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
plt.title("Logistic Regression decision boundary")
plt.show()
