from collections import defaultdict
import math

class enaiveBayes:
    def fit(self, X, y):
        self.classes = set(y)
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(int) 
        for xi, yi in zip(X, y):
            self.class_counts[yi] += 1
            for j, val in enumerate(xi):
                self.feature_counts[(yi, j, val)] += 1  

        self.total_samples = len(y)

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                log_prob = math.log(self.class_counts[c] / self.total_samples)
                for j, val in enumerate(x):
                    count = self.feature_counts[(c, j, val)] + 1 
                    total = self.class_counts[c] + len(self.classes)
                    log_prob += math.log(count / total)
                class_probs[c] = log_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions
