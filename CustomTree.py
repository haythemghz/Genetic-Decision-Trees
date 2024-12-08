import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy
import random
from Node import Node

class CustomTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        if len(y) == 0 or len(set(y)) == 1 or depth >= self.max_depth:
            return Node(value=np.mean(y) if len(y) > 0 else 0)

        feature_index, threshold = self._find_best_split(X, y)
        if feature_index is None:
            return Node(value=np.mean(y))

        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        impurities = []
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                impurity = self._calculate_impurity(y[left_indices], y[right_indices])
                impurities.append((feature_index, threshold, impurity))

        # Select the split with the lowest impurity
        best_feature, best_threshold, best_impurity = min(impurities, key=lambda x: x[2])
        return best_feature, best_threshold

    def _calculate_impurity(self, left, right):
        total_length = len(left) + len(right)
        if total_length == 0:
            return float('inf')
        p_left = len(left) / total_length
        p_right = len(right) / total_length
        return p_left * self._gini(left) + p_right * self._gini(right)

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        return np.array([1 if self._traverse_tree(self.root, x) >= 0.5 else 0 for x in X])

    def _traverse_tree(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(node.left, x)
        else:
            return self._traverse_tree(node.right, x)

    def fitness(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        impurity = self._calculate_tree_impurity(self.root, X, y)
        # Avoid division by zero by adding a small constant
        epsilon = 1e-10
        return accuracy - impurity + epsilon

    def _calculate_tree_impurity(self, node, X, y):
        if node is None or node.is_leaf():
            return self._gini(y)
        left_indices = X[:, node.feature_index] <= node.threshold
        right_indices = ~left_indices
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return self._gini(y)
        left_impurity = self._calculate_tree_impurity(node.left, X[left_indices], y[left_indices])
        right_impurity = self._calculate_tree_impurity(node.right, X[right_indices], y[right_indices])
        return (len(y[left_indices]) * left_impurity + len(y[right_indices]) * right_impurity) / len(y)

    def prune(self, X, y):
        self._prune_node(self.root, X, y)

    def _prune_node(self, node, X, y):
        if node is None or node.is_leaf():
            return

        if node.left:
            left_indices = X[:, node.feature_index] <= node.threshold
            self._prune_node(node.left, X[left_indices], y[left_indices])

        if node.right:
            right_indices = X[:, node.feature_index] > node.threshold
            self._prune_node(node.right, X[right_indices], y[right_indices])

        if node.left and node.left.is_leaf() and node.right and node.right.is_leaf():
            node_value = np.mean(y)
            binary_predictions = np.array([1 if val >= 0.5 else 0 for val in [node_value] * len(y)])
            pruned_fitness = accuracy_score(y, binary_predictions)

            # Ensure original predictions are binary
            original_predictions = np.array([1 if node.left.value >= 0.5 else 0 if x[node.feature_index] <= node.threshold
                                              else 1 if node.right.value >= 0.5 else 0 for x in X])
            original_fitness = accuracy_score(y, original_predictions)

            if pruned_fitness >= original_fitness:
                node.value = node_value
                node.left = None
                node.right = None
