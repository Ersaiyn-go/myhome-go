class ePCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = []
        self.mean = []

    def fit(self, X):
        n = len(X)
        m = len(X[0])
        self.mean = [sum(x[i] for x in X) / n for i in range(m)]
        centered = [[x[i] - self.mean[i] for i in range(m)] for x in X]

        # Матрица ковариации
        cov = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)

        # Собственные вектора (приближённо: диагональ ковариации — дисперсия)
        variances = [(i, cov[i][i]) for i in range(m)]
        variances.sort(key=lambda x: x[1], reverse=True)
        self.components = [v[0] for v in variances[:self.n_components]]

    def transform(self, X):
        return [[x[i] - self.mean[i] for i in self.components] for x in X]
