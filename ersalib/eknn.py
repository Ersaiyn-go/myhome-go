from collections import Counter
from math import sqrt

class ekNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x1, x2):
        return sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [(self._distance(x, x_train), y) for x_train, y in zip(self.X_train, self.y_train)]
            k_nearest = sorted(distances, key=lambda x: x[0])[:self.k]
            votes = [label for _, label in k_nearest]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions
