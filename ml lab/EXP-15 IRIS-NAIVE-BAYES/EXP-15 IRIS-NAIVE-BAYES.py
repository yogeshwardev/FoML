from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2,
                                      random_state=1, stratify=iris.target)
nb = GaussianNB().fit(Xtr, ytr)
pred = nb.predict(Xte)

print("Accuracy:", accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=iris.target_names))

sample = [[6.0, 2.9, 4.5, 1.5]]
print("Prediction for", sample, "->", iris.target_names[nb.predict(sample)[0]])
print("Class probabilities:", nb.predict_proba(sample).round(3))
