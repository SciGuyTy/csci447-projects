from typing import List

from Project04.Algorithms.Crossover.Crossover import Crossover
from Project04.NeuralNetwork import NeuralNetwork


class UniformCrossover(Crossover):
    def __init__(self, parent_one: NeuralNetwork, parent_two: NeuralNetwork):
        super().__init__(parent_one, parent_two)

    def cross(self) -> List:
        

        pass
