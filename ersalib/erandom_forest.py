import random
from ersalib import edecisionTreeClassifier

class erandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            indices = random.choices(range(len(X)), k=len(X))
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            tree = edecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return [sum(preds) / len(preds) for preds in zip(*predictions)]
