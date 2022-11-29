import random
from typing import List, Callable

from Project04.Algorithms.Selection.Selection import Selection
from Project04.NeuralNetwork import NeuralNetwork


class TournamentSelect(Selection):
    def __init__(self, fitness_function: Callable):
        super().__init__(fitness_function)

    def select(self, population: List[NeuralNetwork], k: int = 3) -> NeuralNetwork:
        # Select random chromosomes from population (without replacement)
        contestants = random.sample(population, k)

        # Compute and compare fitness
        contestant_fitness = [self.fitness_function(contestant) for contestant in contestants]

        # Find index of contestant with best fitness
        best_fitness = min(contestant_fitness)
        index_of_selection = contestant_fitness.index(best_fitness)

        # Return the best contestant
        return contestants[index_of_selection]
