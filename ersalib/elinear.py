class elinearRegression:
    def __init__(self):
        self.coef_ = []
        self.intercept_ = 0

    def fit(self, X, y):
        n = len(X)
        m = len(X[0])
        x_means = [sum(X[i][j] for i in range(n)) / n for j in range(m)]
        y_mean = sum(y) / n
        self.coef_ = []
        for j in range(m):
            num = sum((X[i][j] - x_means[j]) * (y[i] - y_mean) for i in range(n))
            den = sum((X[i][j] - x_means[j]) ** 2 for i in range(n))
            self.coef_.append(num / den)
        self.intercept_ = y_mean - sum(self.coef_[j] * x_means[j] for j in range(m))

    def predict(self, X):
        return [sum(self.coef_[j] * x[j] for j in range(len(x))) + self.intercept_ for x in X]