from random import sample
from math import dist

class eKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def fit(self, X):
        self.centroids = sample(X, self.k)
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]
            for x in X:
                dists = [dist(x, c) for c in self.centroids]
                cluster_idx = dists.index(min(dists))
                clusters[cluster_idx].append(x)
            new_centroids = [self._mean(cluster) for cluster in clusters]
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
        self.labels_ = [self._predict_one(x) for x in X]

    def _mean(self, points):
        return [sum(x[i] for x in points) / len(points) for i in range(len(points[0]))]

    def _predict_one(self, x):
        return min(range(self.k), key=lambda i: dist(x, self.centroids[i]))

    def predict(self, X):
        return [self._predict_one(x) for x in X]
