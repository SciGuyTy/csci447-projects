import numpy as np
from typing import List, Tuple

class Mutation():
    def __init__(self, mutation_range: tuple[float, float], probability_of_mutation: float):
        self.mutation_range = mutation_range
        self.probability_of_mutation = probability_of_mutation

    def mutate(self, chromosomes: List[List[float]]) -> List[List[float]]:
        pass
