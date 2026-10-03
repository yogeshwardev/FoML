import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)


def knn_predict(Xtrain, ytrain, x, k):
    dist = np.sqrt(((Xtrain - x) ** 2).sum(axis=1))
    nearest = ytrain[np.argsort(dist)[:k]]
    return Counter(nearest).most_common(1)[0][0]


k = 5
pred = [knn_predict(Xtr, ytr, x, k) for x in Xte]
print(f"From-scratch KNN (k={k}) accuracy:", accuracy_score(yte, pred))

sk = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
print("sklearn KNN accuracy            :", accuracy_score(yte, sk.predict(Xte)))

for k in (1, 3, 5, 7, 9):
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    print(f"k={k}: accuracy = {m.score(Xte, yte):.4f}")
