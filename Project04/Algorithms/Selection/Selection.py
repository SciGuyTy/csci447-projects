from typing import List, Callable

from Project04.NeuralNetwork import NeuralNetwork


class Selection:
    def __init__(self, fitness_function: Callable):
        self.fitness_function = fitness_function

    def select(self, population: List[NeuralNetwork]) -> List:
        pass
