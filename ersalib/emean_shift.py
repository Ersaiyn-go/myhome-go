from math import dist

class eMeanShift:
    def __init__(self, radius=2.0, max_iters=100):
        self.radius = radius
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[:]
        for _ in range(self.max_iters):
            new_centroids = []
            for c in self.centroids:
                in_band = [x for x in X if dist(x, c) <= self.radius]
                if in_band:
                    new_centroids.append(self._mean(in_band))
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
        self.labels_ = [self._closest(x) for x in X]

    def _mean(self, points):
        return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]

    def _closest(self, x):
        return min(range(len(self.centroids)), key=lambda i: dist(x, self.centroids[i]))

    def predict(self, X):
        return [self._closest(x) for x in X]
