import numpy as np
from typing import List, Tuple

from Project04.NeuralNetwork import NeuralNetwork


class Crossover():
    def __init__(self, parent_one: NeuralNetwork, parent_two: NeuralNetwork, probability_of_cross: float = 0.8):
        self.parent_one = parent_one
        self.parent_two = parent_two
        self.probability_of_cross = probability_of_cross

    def cross(self) -> List:
        pass
