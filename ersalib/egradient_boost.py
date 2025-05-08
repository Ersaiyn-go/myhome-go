
class SimpleTreeRegressor:
    def __init__(self, depth=2):
        self.depth = depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=self.depth)

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return sum(y) / len(y)

        best_feat, best_thresh, best_mse = 0, 0, float('inf')
        best_split = ([], [])
        for feat in range(len(X[0])):
            values = set(x[feat] for x in X)
            for thresh in values:
                left_idx = [i for i in range(len(X)) if X[i][feat] < thresh]
                right_idx = [i for i in range(len(X)) if X[i][feat] >= thresh]
                if not left_idx or not right_idx:
                    continue
                left = [y[i] for i in left_idx]
                right = [y[i] for i in right_idx]
                mse = self._mse(left, right)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh
                    best_split = (left_idx, right_idx)

        left_tree = self._build_tree([X[i] for i in best_split[0]], [y[i] for i in best_split[0]], depth - 1)
        right_tree = self._build_tree([X[i] for i in best_split[1]], [y[i] for i in best_split[1]], depth - 1)

        return {
            "feature": best_feat,
            "threshold": best_thresh,
            "left": left_tree,
            "right": right_tree
        }

    def _mse(self, left, right):
        def mean(lst): return sum(lst) / len(lst) if lst else 0
        left_m, right_m = mean(left), mean(right)
        return sum((l - left_m) ** 2 for l in left) + sum((r - right_m) ** 2 for r in right)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] < node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return [self.predict_one(x, self.tree) for x in X]



class eGradientBoosting:
    def __init__(self, n_estimators=50, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.models = []

    def fit(self, X, y):
        n = len(y)
        pred = [0.0] * n
        self.models = []

        for _ in range(self.n_estimators):
            residual = [y[i] - pred[i] for i in range(n)]
            tree = SimpleTreeRegressor(depth=2)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred = [pred[i] + self.lr * update[i] for i in range(n)]
            self.models.append(tree)

    def predict(self, X):
        pred = [0.0] * len(X)
        for model in self.models:
            update = model.predict(X)
            pred = [pred[i] + self.lr * update[i] for i in range(len(X))]
        return pred
