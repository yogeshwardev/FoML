import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(1)
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=1)


def sigmoid(z): return 1 / (1 + np.exp(-z))
def dsigmoid(a): return a * (1 - a)


n_in, n_hid, n_out = 4, 8, 3
W1 = np.random.randn(n_in, n_hid) * 0.5
b1 = np.zeros((1, n_hid))
W2 = np.random.randn(n_hid, n_out) * 0.5
b2 = np.zeros((1, n_out))
lr, epochs = 0.1, 2000

for ep in range(1, epochs + 1):
    h = sigmoid(Xtr @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    err = Ytr - out
    d_out = err * dsigmoid(out)
    d_hid = (d_out @ W2.T) * dsigmoid(h)
    W2 += lr * h.T @ d_out / len(Xtr)
    b2 += lr * d_out.mean(axis=0, keepdims=True)
    W1 += lr * Xtr.T @ d_hid / len(Xtr)
    b1 += lr * d_hid.mean(axis=0, keepdims=True)
    if ep % 400 == 0:
        print(f"Epoch {ep:4d}  MSE = {np.mean(err ** 2):.5f}")

pred = sigmoid(sigmoid(Xte @ W1 + b1) @ W2 + b2)
acc = np.mean(pred.argmax(1) == Yte.argmax(1))
print("Test accuracy:", round(acc, 4))
