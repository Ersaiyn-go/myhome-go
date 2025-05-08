class eSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        y = [1 if i == 1 else -1 for i in y]  # Ensure labels are -1 or 1
        n = len(X)
        m = len(X[0])
        self.weights = [0.0] * m
        self.bias = 0.0

        for _ in range(self.epochs):
            for i in range(n):
                linear_output = sum(self.weights[j] * X[i][j] for j in range(m)) + self.bias
                if y[i] * linear_output < 1:
                    for j in range(m):
                        self.weights[j] += self.lr * (y[i] * X[i][j] - self.lambda_param * self.weights[j])
                    self.bias += self.lr * y[i]
                else:
                    for j in range(m):
                        self.weights[j] += self.lr * (-self.lambda_param * self.weights[j])

    def predict(self, X):
        preds = []
        for x in X:
            linear_output = sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias
            preds.append(1 if linear_output >= 0 else 0)
        return preds