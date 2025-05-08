class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class edecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return Node(value=sum(y) / len(y))

        best_feature, best_thresh, best_gain = 0, 0, float('-inf')
        for feature in range(len(X[0])):
            thresholds = set([x[feature] for x in X])
            for thresh in thresholds:
                left_idx = [i for i in range(len(X)) if X[i][feature] < thresh]
                right_idx = [i for i in range(len(X)) if X[i][feature] >= thresh]
                if not left_idx or not right_idx:
                    continue
                left = [y[i] for i in left_idx]
                right = [y[i] for i in right_idx]
                gain = self._variance_reduction(y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = thresh
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        left = self._build_tree([X[i] for i in best_left_idx], [y[i] for i in best_left_idx], depth + 1)
        right = self._build_tree([X[i] for i in best_right_idx], [y[i] for i in best_right_idx], depth + 1)
        return Node(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def _variance(self, y):
        mean_y = sum(y) / len(y)
        return sum((yi - mean_y) ** 2 for yi in y) / len(y)

    def _variance_reduction(self, parent, left, right):
        return self._variance(parent) - (
            len(left) / len(parent)) * self._variance(left) - (
            len(right) / len(parent)) * self._variance(right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.root) for x in X]