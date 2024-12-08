import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from copy import deepcopy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from graphviz import Digraph
from GeneticAlgorithm import GeneticAlgorithm
from CustomTree import CustomTree

# Load dataset and prepare data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, y_test = X_val, y_val  # Use the same validation set as the test set for simplicity

# Run Genetic Algorithm
ga = GeneticAlgorithm(
    population_size=200,
    generations=20,
    mutation_rate=0.5,
    crossover_rate=0.8,
    max_depth=15,
    pruning_probability=0.1  # Low probability for selective pruning
)
best_tree = ga.run(X_train, y_train, X_val, y_val)

# Evaluate the best tree on the test set
test_predictions = best_tree.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
confusion = confusion_matrix(y_test, test_predictions)

print(f"Accuracy of the best tree on the test set: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")

# Function to draw the decision tree using Graphviz
def draw_tree(tree, feature_names, class_names):
    dot = Digraph()
    _add_nodes_edges(dot, tree.root, feature_names, class_names)
    return dot

def _add_nodes_edges(dot, node, feature_names, class_names):
    if node.is_leaf():
        dot.node(str(id(node)), label=f"Value: {node.value:.2f}")
        return

    feature_name = feature_names[node.feature_index]
    threshold = node.threshold
    dot.node(str(id(node)), label=f"{feature_name} <= {threshold:.2f}")

    if node.left:
        dot.edge(str(id(node)), str(id(node.left)), label="True")
        _add_nodes_edges(dot, node.left, feature_names, class_names)

    if node.right:
        dot.edge(str(id(node)), str(id(node.right)), label="False")
        _add_nodes_edges(dot, node.right, feature_names, class_names)

# Draw the best tree
feature_names = data.columns[:-1]
class_names = ['0', '1']
tree_dot = draw_tree(best_tree, feature_names, class_names)
tree_dot.render('best_tree', format='png', view=True)
