from typing import List, Callable

from Project04.NeuralNetwork import NeuralNetwork


class Selection:
    def __init__(self, population: List[NeuralNetwork], fitness_function: Callable):
        self.population = population
        self.fitness_function = fitness_function

    def select(self) -> List:
        pass
