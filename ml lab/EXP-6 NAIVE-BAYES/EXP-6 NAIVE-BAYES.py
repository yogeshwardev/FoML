from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

model = GaussianNB().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Confusion matrix:\n", confusion_matrix(yte, pred))
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print("\n", classification_report(yte, pred))
