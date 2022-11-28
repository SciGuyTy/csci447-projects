import numpy as np
from typing import List, Tuple

class Crossover():
    def __init__(self, probability_of_cross: float):
        self.probability_of_cross = probability_of_cross

    def cross(self, parent_one: List[float], parent_two: List[float]) -> List:
        pass
