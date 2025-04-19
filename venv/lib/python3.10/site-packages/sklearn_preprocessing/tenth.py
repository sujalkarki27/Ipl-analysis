def print_prog():
    print(
        """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, title):
    history = model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    plt.plot(history.loss_curve_, label=title)
    return accuracy

 

supervised_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
supervised_accuracy = train_and_evaluate(supervised_model, X_train, y_train, X_test, y_test, "Supervised")

 

unsupervised_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1, warm_start=True, random_state=42)

 

unsupervised_loss = []
for _ in range(100):
    unsupervised_model.fit(X_train, y_train)
    unsupervised_loss.append(unsupervised_model.loss_)

unsupervised_accuracy = unsupervised_model.score(X_test, y_test)
plt.plot(unsupervised_loss, label="Unsupervised")

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"Supervised Accuracy: {supervised_accuracy}")
print(f"Unsupervised Accuracy: {unsupervised_accuracy}")
"""
    )
