import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(2)
n = 2000
df = pd.DataFrame({
    "battery_power": rng.integers(500, 2000, n),
    "ram": rng.integers(256, 4000, n),
    "int_memory": rng.integers(2, 64, n),
    "px_height": rng.integers(0, 1960, n),
    "px_width": rng.integers(500, 1998, n),
    "mobile_wt": rng.integers(80, 200, n),
    "n_cores": rng.integers(1, 9, n),
    "fc": rng.integers(0, 20, n),
})
score = df.ram / 1000 + df.battery_power / 1000 + df.px_width / 1000 \
    + df.int_memory / 30 + rng.normal(0, 0.5, n)
df["price_range"] = pd.qcut(score, 4, labels=[0, 1, 2, 3]).astype(int)

X, y = df.drop(columns="price_range"), df["price_range"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print(classification_report(yte, pred))
print("Predicted price range for first test phone:", model.predict(Xte.iloc[[0]])[0])
