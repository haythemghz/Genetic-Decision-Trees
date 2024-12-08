import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy
import random
from CustomTree import CustomTree

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, max_depth, pruning_probability):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.pruning_probability = pruning_probability

    def initialize_population(self, X, y):
        population = [CustomTree(max_depth=self.max_depth) for _ in range(self.population_size)]
        for tree in population:
            tree.fit(X, y)
        return population

    def evaluate_population(self, population, X, y):
        return [tree.fitness(X, y) for tree in population]

    def select_parents(self, population, fitness_scores):
        # Tournament selection
        tournament_size = 3
        selected = random.choices(population, weights=fitness_scores, k=tournament_size)
        best_parent = max(selected, key=lambda tree: tree.fitness(X_val, y_val))
        return best_parent, best_parent

    def mutate(self, tree, X, y):
        new_tree = deepcopy(tree)
        new_tree.prune(X, y)
        return new_tree

    def crossover(self, parent1, parent2):
        child = deepcopy(parent1)
        if random.random() < self.crossover_rate:
            donor_subtree = self._select_random_subtree(parent2.root)
            recipient_subtree = self._select_random_subtree(child.root)
            if donor_subtree and recipient_subtree:
                recipient_subtree.feature_index = donor_subtree.feature_index
                recipient_subtree.threshold = donor_subtree.threshold
                recipient_subtree.left = donor_subtree.left
                recipient_subtree.right = donor_subtree.right
        return child

    def _select_random_subtree(self, node):
        if node is None or node.is_leaf():
            return node
        if random.random() < 0.5:
            return self._select_random_subtree(node.left)
        else:
            return self._select_random_subtree(node.right)

    def run(self, X_train, y_train, X_val, y_val):
        population = self.initialize_population(X_train, y_train)

        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            fitness_scores = self.evaluate_population(population, X_val, y_val)
            print(f"Best fitness in generation: {max(fitness_scores)}")

            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child, X_train, y_train)
                new_population.append(child)

            population = new_population

            # Apply selective pruning as a genetic operator
            pruned_trees = []
            for tree in population:
                if random.random() < self.pruning_probability:
                    pruned_tree = deepcopy(tree)
                    pruned_tree.prune(X_train, y_train)
                    pruned_trees.append(pruned_tree)

            population.extend(pruned_trees)

        best_tree_index = np.argmax(self.evaluate_population(population, X_val, y_val))
        return population[best_tree_index]
