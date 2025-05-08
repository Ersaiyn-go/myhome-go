import math

class elogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n = len(X)
        m = len(X[0])
        self.weights = [0.0] * m
        self.bias = 0.0

        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.weights[j] * X[i][j] for j in range(m)) + self.bias
                pred = self.sigmoid(z)
                error = y[i] - pred
                for j in range(m):
                    self.weights[j] += self.lr * error * X[i][j]
                self.bias += self.lr * error

    def predict(self, X):
        preds = []
        for x in X:
            z = sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias
            preds.append(1 if self.sigmoid(z) >= 0.5 else 0)
        return preds