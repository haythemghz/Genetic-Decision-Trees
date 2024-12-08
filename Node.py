class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Value for leaf nodes (mean of target values)

    def is_leaf(self):
        return self.value is not None
