import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

yb = (y == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=0.2, random_state=0)
w, b, lr = np.zeros(X.shape[1]), 0.0, 0.1
for epoch in range(20):
    errors = 0
    for xi, ti in zip(Xtr, ytr):
        o = int(xi @ w + b > 0)
        upd = lr * (ti - o)
        w += upd * xi
        b += upd
        errors += int(upd != 0)
    if errors == 0:
        print(f"Converged at epoch {epoch + 1}")
        break
pred = (Xte @ w + b > 0).astype(int)
print("Scratch perceptron (Setosa vs rest) accuracy:", accuracy_score(yte, pred))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
clf = Perceptron(max_iter=1000, random_state=0).fit(sc.transform(Xtr), ytr)
pred = clf.predict(sc.transform(Xte))
print("\nsklearn Perceptron (3 classes) accuracy:", accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
