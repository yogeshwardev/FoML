import math
import pandas as pd

data = pd.DataFrame({
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                    "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                 "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
             "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                   "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
})
TARGET = "PlayTennis"


def entropy(col):
    probs = col.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs)


def info_gain(df, attr):
    rem = sum(len(s) / len(df) * entropy(s[TARGET]) for _, s in df.groupby(attr))
    return entropy(df[TARGET]) - rem


def id3(df, attrs):
    if df[TARGET].nunique() == 1:
        return df[TARGET].iloc[0]
    if not attrs:
        return df[TARGET].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a))
    tree = {best: {}}
    for val, sub in df.groupby(best):
        tree[best][val] = id3(sub, [a for a in attrs if a != best])
    return tree


def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    return classify(tree[attr][sample[attr]], sample)


def show(tree, indent=0):
    if not isinstance(tree, dict):
        print(" " * indent + "->", tree)
        return
    for attr, branches in tree.items():
        for val, sub in branches.items():
            print(" " * indent + f"{attr} = {val}")
            show(sub, indent + 4)


features = [c for c in data.columns if c != TARGET]
print("Entropy of dataset:", round(entropy(data[TARGET]), 4))
for f in features:
    print(f"Info gain({f}) = {info_gain(data, f):.4f}")

tree = id3(data, features)
print("\nDecision tree:")
show(tree)

new = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
print("\nNew sample:", new)
print("Prediction:", classify(tree, new))
