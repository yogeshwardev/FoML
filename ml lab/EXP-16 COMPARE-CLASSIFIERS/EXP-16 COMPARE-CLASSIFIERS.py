from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "SVM": SVC(),
}

print(f"{'Model':20s}{'Acc':>7s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'CV-5':>7s}")
accs = {}
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m)
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    accs[name] = accuracy_score(yte, p)
    cv = cross_val_score(pipe, X, y, cv=5).mean()
    print(f"{name:20s}{accs[name]:7.3f}{precision_score(yte, p):7.3f}"
          f"{recall_score(yte, p):7.3f}{f1_score(yte, p):7.3f}{cv:7.3f}")

plt.barh(list(accs), list(accs.values()))
plt.xlabel("Test accuracy"); plt.title("Classifier comparison"); plt.tight_layout(); plt.show()
